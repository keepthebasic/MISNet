from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .misnet import MISNetConfig, MISNetLossWeights, misnet_loss, masked_mse
from .misnet_blocks import CoupledSpecMoE, HyCoPeriodNet, LRMaskFormer, SLAReadout


class ChlaPersistResidualAMS(nn.Module):

    def __init__(self, d_model: int, balance_coef: float = 0.02):
        super().__init__()
        self.balance_coef = float(balance_coef)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias[0], -0.5)
        nn.init.constant_(self.gate[-1].bias[1], 0.5)

    def forward(self, ctx: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(ctx)
        gates = F.softmax(logits, dim=-1)
        gated = gates[:, 1:2] * delta
        mean_g = gates.mean(dim=0)
        balance = self.balance_coef * ((mean_g - mean_g.new_tensor([0.5, 0.5])).pow(2).sum())
        return gated, balance


class ChlaGapDecayGate(nn.Module):

    def __init__(self, d_model: int, max_age: float = 48.0):
        super().__init__()
        self.max_age = float(max_age)
        self.gate = nn.Sequential(
            nn.Linear(d_model + 4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 1.5)

    @staticmethod
    def time_since_obs(mask: torch.Tensor) -> torch.Tensor:
        b, c, length = mask.shape
        ages = mask.new_zeros(b, c, length)
        for t in range(1, length):
            ages[:, :, t] = (1.0 - mask[:, :, t]) * (ages[:, :, t - 1] + 1.0)
        return ages

    def forward(self, ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ages = self.time_since_obs(mask)
        last_age = ages[:, :, -1] / self.max_age
        mean_age = last_age.mean(dim=1, keepdim=True)
        max_age = last_age.max(dim=1, keepdim=True).values
        L = mask.size(-1)
        k = min(24, L)
        recent = mask[:, :, -k:].mean(dim=(1, 2)).unsqueeze(-1)
        last_rate = mask[:, :, -1].mean(dim=1, keepdim=True)
        feats = torch.cat([mean_age, max_age, recent, last_rate], dim=-1)
        logits = self.gate(torch.cat([ctx, feats], dim=-1))
        return torch.sigmoid(logits)


class DualScaleGated1D(nn.Module):

    def __init__(self, dim: int, expand: float = 2.0):
        super().__init__()
        hidden = max(dim, int(dim * expand))
        self.project_in = nn.Conv1d(dim, hidden * 2, kernel_size=1)
        self.dw_wide = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.dw_dilated = nn.Conv1d(
            hidden, hidden, kernel_size=3, padding=2, groups=hidden, dilation=2
        )
        self.project_out = nn.Conv1d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.project_in(x).chunk(2, dim=1)
        y = F.mish(self.dw_dilated(b)) * self.dw_wide(a)
        return self.project_out(y)


class HighLowGatedMix1D(nn.Module):

    def __init__(self, dim: int, low_kernel: int = 7):
        super().__init__()
        k = low_kernel if low_kernel % 2 == 1 else low_kernel + 1
        self.low_pool = nn.AvgPool1d(kernel_size=k, stride=1, padding=k // 2)
        self.low_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.high_ffn = DualScaleGated1D(dim)
        self.gate = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_pool(x)
        if low.size(-1) != x.size(-1):
            low = F.interpolate(low, size=x.size(-1), mode="linear", align_corners=False)
        high = x - low
        low_f = self.low_proj(low)
        high_f = self.high_ffn(high)
        g = self.gate(torch.cat([low_f, high_f], dim=1))
        y = g * high_f + (1.0 - g) * low_f
        return self.norm(x + y)


class ChannelIndependentChlaHead(nn.Module):

    def __init__(self, in_channels: int, pred_len: int, hidden: int = 32, kernel: int = 7):
        super().__init__()
        k = kernel if kernel % 2 == 1 else kernel + 1
        self.per_ch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, hidden, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for _ in range(in_channels)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * in_channels, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, pred_len),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feats = []
        for c, enc in enumerate(self.per_ch):
            xc = torch.nan_to_num(x[:, c : c + 1], nan=0.0) * mask[:, c : c + 1]
            feats.append(enc(xc).squeeze(-1))
        return self.fuse(torch.cat(feats, dim=-1))


@dataclass
class MISNetHFDConfig(MISNetConfig):
    n_hfd_blocks: int = 2
    low_kernel: int = 7
    hard_do_persistence: bool = True
    chla_anchor_mode: str = "obs_if_valid"
    use_hydro_periodic: bool = False
    use_spectral_mixer: bool = False
    chla_loss_weight: float = 1.5
    chla_horizon_weights: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1, 1.25)
    use_chla_ci: bool = False
    chla_ci_hidden: int = 32
    use_chla_ams: bool = False
    chla_ams_balance: float = 0.02
    use_chla_gap_decay: bool = False
    chla_decay_max_age: float = 48.0
    do_horizon_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class MISNetHFD(nn.Module):
    def __init__(self, cfg: MISNetHFDConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            base = {k: v for k, v in kwargs.items() if k in MISNetHFDConfig.__dataclass_fields__}
            cfg = MISNetHFDConfig(**base)
        self.cfg = cfg

        self.missingness_encoder = LRMaskFormer(
            cfg.in_channels,
            cfg.d_model,
            mute_dead_channels=cfg.mute_dead_channels,
            dead_channel_thresh=cfg.dead_channel_thresh,
            enhance_per_var=cfg.enhance_per_var,
        )
        self.hydro_periodic: nn.Module = (
            HyCoPeriodNet(cfg.d_model, cfg.n_period_bands)
            if cfg.use_hydro_periodic
            else nn.Identity()
        )
        self.spectral_mixer: CoupledSpecMoE | None = (
            CoupledSpecMoE(
                cfg.d_model,
                n_bands=cfg.n_spec_bands,
                rank=cfg.cross_rank,
                dropout=cfg.dropout,
            )
            if cfg.use_spectral_mixer
            else None
        )
        self.blocks = nn.ModuleList(
            [HighLowGatedMix1D(cfg.d_model, low_kernel=cfg.low_kernel) for _ in range(cfg.n_hfd_blocks)]
        )
        self.lab_readout = (
            SLAReadout(
                cfg.d_model,
                cfg.in_channels,
                mute_dead_channels=cfg.mute_dead_channels,
                dead_channel_thresh=cfg.dead_channel_thresh,
            )
            if cfg.use_sla
            else None
        )
        self.ctx_pool = nn.AdaptiveAvgPool1d(1)
        self.readout_gate = nn.Sequential(nn.Linear(cfg.d_model * 2, cfg.d_model), nn.Sigmoid())
        self.soft_prior = nn.Parameter(torch.zeros(cfg.n_soft))
        self.soft_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_soft),
        )
        self.chla_delta_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.pred_len),
        )
        nn.init.zeros_(self.chla_delta_head[-1].weight)
        nn.init.zeros_(self.chla_delta_head[-1].bias)
        nn.init.zeros_(self.soft_head[-1].weight)
        nn.init.zeros_(self.soft_head[-1].bias)

        self.chla_ci: ChannelIndependentChlaHead | None = None
        self.chla_mix_gate: nn.Module | None = None
        self.chla_ams: ChlaPersistResidualAMS | None = None
        self.chla_gap_decay: ChlaGapDecayGate | None = None
        if cfg.use_chla_ci:
            self.chla_ci = ChannelIndependentChlaHead(
                cfg.in_channels, cfg.pred_len, hidden=cfg.chla_ci_hidden
            )
            self.chla_mix_gate = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.chla_mix_gate[-2].bias, 1.0)
        if cfg.use_chla_ams:
            self.chla_ams = ChlaPersistResidualAMS(
                cfg.d_model, balance_coef=cfg.chla_ams_balance
            )
        if cfg.use_chla_gap_decay:
            self.chla_gap_decay = ChlaGapDecayGate(
                cfg.d_model, max_age=cfg.chla_decay_max_age
            )

    def _resolve_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            mask = (~torch.isnan(x)).float()
        return mask

    def _chla_anchor(
        self,
        soft: torch.Tensor,
        soft_y: Optional[torch.Tensor],
        soft_y_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        chla_i = min(max(self.cfg.chla_soft_index, 0), soft.size(1) - 1)
        mode = (self.cfg.chla_anchor_mode or "obs_if_valid").lower()
        soft_pred = soft[:, chla_i].detach()
        if mode == "zero":
            return torch.zeros(soft.size(0), device=soft.device, dtype=soft.dtype)
        if mode == "obs_if_valid" and soft_y is not None and soft_y_mask is not None:
            obs = torch.nan_to_num(soft_y[:, chla_i], nan=0.0)
            m = soft_y_mask[:, chla_i].float()
            return m * obs + (1.0 - m) * soft_pred
        return soft_pred

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        exo: Optional[torch.Tensor] = None,
        soft_y: Optional[torch.Tensor] = None,
        soft_y_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        del exo
        cfg = self.cfg
        mask = self._resolve_mask(x, mask)
        h = self.missingness_encoder(x, mask if cfg.use_mask else torch.ones_like(mask))
        h = self.hydro_periodic(h)
        balance = h.new_zeros(())
        if self.spectral_mixer is not None:
            h, _sync, _fc, spec_bal = self.spectral_mixer(h)
            balance = balance + spec_bal
        for blk in self.blocks:
            h = blk(h)

        last = h[:, :, -1]
        pooled = self.ctx_pool(h).squeeze(-1)
        if self.lab_readout is not None:
            sla = self.lab_readout(h, mask)
        else:
            sla = pooled
        g = self.readout_gate(torch.cat([last, sla], dim=-1))
        ctx = g * last + (1.0 - g) * sla

        soft = self.soft_prior.view(1, -1) + self.soft_head(ctx)

        do_i = min(max(cfg.do_channel_index, 0), x.size(1) - 1)
        do_last = torch.nan_to_num(x[:, do_i, -1], nan=0.0) * mask[:, do_i, -1]
        do_traj = do_last.unsqueeze(-1).expand(-1, cfg.pred_len)

        chla_t0 = self._chla_anchor(soft, soft_y, soft_y_mask)
        ctx_delta = self.chla_delta_head(ctx)
        if self.chla_ci is not None and self.chla_mix_gate is not None:
            ci_delta = self.chla_ci(x, mask)
            gate = self.chla_mix_gate(ctx)
            chla_delta = gate * ci_delta + (1.0 - gate) * ctx_delta
        else:
            chla_delta = ctx_delta
        if self.chla_ams is not None:
            chla_delta, ams_bal = self.chla_ams(ctx, chla_delta)
            balance = balance + ams_bal
        if self.chla_gap_decay is not None:
            decay_g = self.chla_gap_decay(ctx, mask)
            chla_delta = decay_g * chla_delta
        chla_traj = chla_t0.unsqueeze(-1) + chla_delta
        forecast = torch.stack([do_traj, chla_traj], dim=1)

        return {
            "soft": soft,
            "forecast": forecast,
            "risk": torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype),
            "moe_loss": balance,
            "hidden": h,
        }


def build_misnet_hfd(**kwargs) -> MISNetHFD:
    kwargs.setdefault("use_mask", True)
    kwargs.setdefault("use_sla", True)
    kwargs.setdefault("mute_dead_channels", True)
    kwargs.setdefault("enhance_per_var", True)
    kwargs.setdefault("hard_do_persistence", True)
    kwargs.setdefault("chla_anchor_mode", "obs_if_valid")
    kwargs.setdefault("use_hydro_periodic", False)
    kwargs.setdefault("use_spectral_mixer", False)
    return MISNetHFD(
        MISNetHFDConfig(
            **{k: v for k, v in kwargs.items() if k in MISNetHFDConfig.__dataclass_fields__}
        )
    )


def build_misnet_hfd_chl(**kwargs) -> MISNetHFD:
    kwargs.setdefault("use_chla_ci", True)
    kwargs.setdefault("chla_loss_weight", 2.5)
    kwargs.setdefault(
        "chla_horizon_weights",
        (0.5, 0.7, 0.9, 1.2, 1.5, 2.0),
    )
    kwargs.setdefault("chla_anchor_mode", "obs_if_valid")
    return build_misnet_hfd(**kwargs)


def build_misnet_hfd_chl_ams(**kwargs) -> MISNetHFD:
    kwargs.setdefault("use_chla_ams", True)
    kwargs.setdefault("chla_ams_balance", 0.02)
    return build_misnet_hfd_chl(**kwargs)


def build_misnet_hfd_chl_decay(**kwargs) -> MISNetHFD:
    kwargs.setdefault("use_chla_gap_decay", True)
    kwargs.setdefault("chla_decay_max_age", 48.0)
    kwargs.setdefault("use_chla_ams", False)
    return build_misnet_hfd_chl(**kwargs)


def build_misnet_hfd_chl_hs(**kwargs) -> MISNetHFD:
    kwargs.setdefault("use_hydro_periodic", True)
    kwargs.setdefault("use_spectral_mixer", True)
    kwargs.setdefault("use_chla_ams", False)
    kwargs.setdefault("use_chla_gap_decay", False)
    return build_misnet_hfd_chl(**kwargs)


def build_misnet_hfd_chl_tune(**kwargs) -> MISNetHFD:
    kwargs.setdefault("use_chla_ci", True)
    kwargs.setdefault("chla_loss_weight", 4.0)
    kwargs.setdefault(
        "chla_horizon_weights",
        (0.25, 0.4, 0.6, 1.0, 1.8, 3.0),
    )
    kwargs.setdefault("chla_anchor_mode", "obs_if_valid")
    kwargs.setdefault("use_chla_ams", False)
    kwargs.setdefault("use_chla_gap_decay", False)
    kwargs.setdefault("use_hydro_periodic", False)
    kwargs.setdefault("use_spectral_mixer", False)
    return build_misnet_hfd(**kwargs)


def misnet_hfd_loss(
    out,
    soft_y,
    forecast_y,
    risk_y=None,
    soft_y_mask=None,
    forecast_y_mask=None,
    weights: MISNetLossWeights | None = None,
    cfg=None,
):
    if weights is None:
        weights = MISNetLossWeights()
    if cfg is None:
        return misnet_loss(
            out, soft_y, forecast_y, risk_y, soft_y_mask, forecast_y_mask, weights, cfg
        )

    if soft_y_mask is not None:
        l_soft = masked_mse(out["soft"], soft_y, soft_y_mask)
    else:
        l_soft = F.mse_loss(out["soft"], torch.nan_to_num(soft_y))

    pred = out["forecast"]
    if forecast_y_mask is None:
        forecast_y_mask = torch.ones_like(pred)

    if getattr(cfg, "hard_do_persistence", False) and pred.size(1) >= 2:
        from .misnet import _horizon_weighted_masked_mse

        h = pred.size(-1)
        ch_w = torch.tensor(cfg.chla_horizon_weights[:h], device=pred.device, dtype=pred.dtype)
        if ch_w.numel() < h:
            ch_w = torch.cat(
                [ch_w, torch.ones(h - ch_w.numel(), device=pred.device, dtype=pred.dtype)]
            )
        ch_w = ch_w / ch_w.mean().clamp(min=1e-6)
        l_forecast = _horizon_weighted_masked_mse(
            pred[:, 1], forecast_y[:, 1], forecast_y_mask[:, 1], ch_w
        )
        l_forecast = float(cfg.chla_loss_weight) * l_forecast
    else:
        return misnet_loss(
            out, soft_y, forecast_y, risk_y, soft_y_mask, forecast_y_mask, weights, cfg
        )

    total = weights.soft * l_soft + weights.forecast * l_forecast + weights.moe_balance * out["moe_loss"]
    return {"soft": l_soft, "forecast": l_forecast, "moe_loss": out["moe_loss"], "total": total}


LIMONHFD = MISNetHFD
LIMONHFDConfig = MISNetHFDConfig
build_limon_hfd = build_misnet_hfd
build_limon_hfd_chl = build_misnet_hfd_chl
build_limon_hfd_chl_ams = build_misnet_hfd_chl_ams
build_limon_hfd_chl_decay = build_misnet_hfd_chl_decay
build_limon_hfd_chl_hs = build_misnet_hfd_chl_hs
build_limon_hfd_chl_tune = build_misnet_hfd_chl_tune
limon_hfd_loss = misnet_hfd_loss
