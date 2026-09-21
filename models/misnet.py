from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .misnet_blocks import (
    CoupledSpecMoE,
    ExogenousGate,
    HyCoPeriodNet,
    LRMaskFormer,
    SLAReadout,
)


@dataclass
class MISNetConfig:
    in_channels: int = 5
    n_exo: int = 2
    seq_len: int = 168
    d_model: int = 64
    n_soft: int = 3
    n_forecast: int = 2
    pred_len: int = 6
    n_period_bands: int = 3
    n_spec_bands: int = 3
    cross_rank: int = 8
    use_mask: bool = True
    use_exo: bool = False
    use_sla: bool = True
    use_missingness_encoder: bool = True
    use_hydro_periodic: bool = True
    use_spectral_mixer: bool = True
    use_last_sla_gate: bool = True
    use_persist_residual: bool = True
    mute_dead_channels: bool = True
    dead_channel_thresh: float = 1e-4
    enhance_per_var: bool = True
    do_channel_index: int = 4
    chla_soft_index: int = 2
    dropout: float = 0.1
    do_near_persistence: bool = False
    do_residual_init: float = 0.05
    chla_anchor_mode: str = "soft_pred"
    chla_loss_weight: float = 1.0
    chla_horizon_weights: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
    do_horizon_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass
class MISNetLossWeights:
    soft: float = 1.0
    forecast: float = 1.0
    risk: float = 0.3
    moe_balance: float = 0.01


class MISNet(nn.Module):

    def __init__(self, cfg: MISNetConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = MISNetConfig(**kwargs)
        self.cfg = cfg

        self.missingness_encoder = (
            LRMaskFormer(
                cfg.in_channels,
                cfg.d_model,
                mute_dead_channels=cfg.mute_dead_channels,
                dead_channel_thresh=cfg.dead_channel_thresh,
                enhance_per_var=cfg.enhance_per_var,
            )
            if cfg.use_missingness_encoder
            else nn.Sequential(
                nn.Conv1d(cfg.in_channels, cfg.d_model, 3, padding=1),
                nn.GELU(),
            )
        )
        self.hydro_periodic = (
            HyCoPeriodNet(cfg.d_model, cfg.n_period_bands)
            if cfg.use_hydro_periodic
            else nn.Identity()
        )
        self.spectral_mixer = (
            CoupledSpecMoE(
                cfg.d_model,
                n_bands=cfg.n_spec_bands,
                rank=cfg.cross_rank,
                dropout=cfg.dropout,
            )
            if cfg.use_spectral_mixer
            else None
        )
        self.lite_refine: nn.Module | None = None
        if (not cfg.use_hydro_periodic) and (not cfg.use_spectral_mixer):
            self.lite_refine = nn.Sequential(
                nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=7, padding=3, groups=cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1),
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

        exo_in = cfg.n_exo if cfg.use_exo else 0
        self.exo_gate = ExogenousGate(cfg.d_model, exo_in) if exo_in else None

        self.readout_gate = (
            nn.Sequential(nn.Linear(cfg.d_model * 2, cfg.d_model), nn.Sigmoid())
            if cfg.use_last_sla_gate
            else None
        )
        self.forecast_ctx_gate = (
            nn.Sequential(nn.Linear(cfg.d_model * 2, cfg.d_model), nn.Sigmoid())
            if cfg.use_last_sla_gate
            else None
        )
        self.soft_prior = nn.Parameter(torch.zeros(cfg.n_soft))

        self.soft_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_soft),
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_forecast * cfg.pred_len),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.n_forecast * cfg.pred_len, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )
        self.do_residual_scale = nn.Parameter(
            torch.tensor(float(cfg.do_residual_init) if cfg.do_near_persistence else 1.0)
        )
        if cfg.use_persist_residual:
            nn.init.zeros_(self.soft_head[-1].weight)
            nn.init.zeros_(self.soft_head[-1].bias)
            nn.init.zeros_(self.forecast_head[-1].weight)
            nn.init.zeros_(self.forecast_head[-1].bias)

    def _resolve_mask(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            mask = (~torch.isnan(x)).float()
        return mask

    def _encode_sequence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if cfg.use_mask:
            enc_mask = mask
        else:
            enc_mask = torch.ones_like(mask)

        if cfg.use_missingness_encoder:
            h = self.missingness_encoder(x, enc_mask)
        else:
            if cfg.mute_dead_channels:
                alive = (enc_mask.mean(dim=-1, keepdim=True) > cfg.dead_channel_thresh).to(
                    dtype=x.dtype
                )
                x = torch.nan_to_num(x, nan=0.0) * alive
            h = self.missingness_encoder(x)

        if self.lite_refine is not None:
            h = h + self.lite_refine(h)

        h = self.hydro_periodic(h)

        if self.spectral_mixer is not None:
            enc, sync_feat, forecast_feat, balance_loss = self.spectral_mixer(h)
        else:
            enc = h
            pooled = self.ctx_pool(h).squeeze(-1)
            sync_feat = forecast_feat = pooled
            balance_loss = torch.zeros((), device=h.device, dtype=h.dtype)

        return enc, sync_feat, forecast_feat, balance_loss

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._resolve_mask(x, mask)
        enc, _, _, _ = self._encode_sequence(x, mask)
        return enc

    def _gated_mix(
        self,
        local: torch.Tensor,
        pooled: torch.Tensor,
        gate: Optional[nn.Module],
    ) -> torch.Tensor:
        if gate is None:
            return pooled
        g = gate(torch.cat([local, pooled], dim=-1))
        return g * local + (1.0 - g) * pooled

    def _chla_anchor(
        self,
        soft: torch.Tensor,
        soft_y: Optional[torch.Tensor],
        soft_y_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.cfg
        chla_i = min(max(cfg.chla_soft_index, 0), soft.size(1) - 1)
        mode = (cfg.chla_anchor_mode or "soft_pred").lower()
        if mode == "zero":
            return torch.zeros(soft.size(0), device=soft.device, dtype=soft.dtype)
        soft_pred = soft[:, chla_i].detach()
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
        cfg = self.cfg
        mask = self._resolve_mask(x, mask)

        enc, sync_feat, forecast_feat, balance_loss = self._encode_sequence(x, mask)
        last = enc[:, :, -1]

        if self.exo_gate is not None and exo is not None:
            sync_feat = self.exo_gate(sync_feat, exo)

        if self.lab_readout is not None:
            sla_feat = self.lab_readout(enc, mask)
        else:
            sla_feat = sync_feat

        soft_ctx = self._gated_mix(last, sla_feat, self.readout_gate)
        fc_ctx = self._gated_mix(last, forecast_feat, self.forecast_ctx_gate)

        delta_soft = self.soft_head(soft_ctx)
        if cfg.use_persist_residual:
            soft = self.soft_prior.view(1, -1) + delta_soft
        else:
            soft = delta_soft

        forecast_flat = self.forecast_head(fc_ctx)
        forecast = forecast_flat.reshape(-1, cfg.n_forecast, cfg.pred_len)
        if cfg.use_persist_residual:
            do_i = min(max(cfg.do_channel_index, 0), x.size(1) - 1)
            do_last = torch.nan_to_num(x[:, do_i, -1], nan=0.0)
            do_last = do_last * mask[:, do_i, -1]
            chla_t0 = self._chla_anchor(soft, soft_y, soft_y_mask)
            do_delta = forecast[:, 0, :]
            if cfg.do_near_persistence:
                scale = torch.nn.functional.softplus(self.do_residual_scale)
                do_delta = scale * do_delta
            parts = [do_delta]
            if cfg.n_forecast > 1:
                parts.append(forecast[:, 1, :])
            delta = torch.stack(parts, dim=1)
            do_anchor = do_last.unsqueeze(-1).expand(-1, cfg.pred_len)
            anchor_parts = [do_anchor]
            if cfg.n_forecast > 1:
                anchor_parts.append(chla_t0.unsqueeze(-1).expand(-1, cfg.pred_len))
            anchor = torch.stack(anchor_parts, dim=1)
            forecast = anchor + delta
            forecast_flat = forecast.reshape(forecast.size(0), -1)

        ctx = self.ctx_pool(enc).squeeze(-1)
        risk_in = torch.cat([ctx, forecast_flat], dim=-1)
        risk = self.risk_head(risk_in)

        return {
            "soft": soft,
            "forecast": forecast,
            "risk": risk,
            "moe_loss": balance_loss,
            "hidden": enc,
        }


def build_misnet(**kwargs) -> MISNet:
    kwargs.setdefault("use_mask", True)
    kwargs.setdefault("use_exo", False)
    kwargs.setdefault("use_sla", True)
    kwargs.setdefault("use_last_sla_gate", True)
    kwargs.setdefault("use_persist_residual", True)
    kwargs.setdefault("mute_dead_channels", True)
    kwargs.setdefault("enhance_per_var", True)
    return MISNet(MISNetConfig(**kwargs))


def build_misnet_forecast_only(**kwargs) -> MISNet:
    return build_misnet(**kwargs)


def build_misnet_rev_l2(**kwargs) -> MISNet:
    kwargs.setdefault("do_near_persistence", True)
    kwargs.setdefault("do_residual_init", 0.05)
    return build_misnet(**kwargs)


def build_misnet_rev_l3(**kwargs) -> MISNet:
    kwargs.setdefault("do_near_persistence", True)
    kwargs.setdefault("do_residual_init", 0.05)
    kwargs.setdefault("chla_anchor_mode", "obs_if_valid")
    kwargs.setdefault("chla_loss_weight", 1.5)
    kwargs.setdefault(
        "chla_horizon_weights",
        (0.9, 0.95, 1.0, 1.05, 1.1, 1.25),
    )
    return build_misnet(**kwargs)


def build_misnet_full(**kwargs) -> MISNet:
    kwargs.setdefault("use_exo", True)
    kwargs.setdefault("n_exo", 2)
    return build_misnet(**kwargs)


def build_misnet_lite(**kwargs) -> MISNet:
    kwargs.setdefault("use_hydro_periodic", False)
    kwargs.setdefault("use_spectral_mixer", False)
    kwargs.setdefault("use_mask", True)
    kwargs.setdefault("use_sla", True)
    kwargs.setdefault("use_last_sla_gate", True)
    kwargs.setdefault("use_persist_residual", True)
    kwargs.setdefault("mute_dead_channels", True)
    kwargs.setdefault("enhance_per_var", True)
    kwargs.setdefault("use_exo", False)
    return MISNet(MISNetConfig(**kwargs))


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.float()
    safe = torch.where(m > 0, target, torch.zeros_like(target))
    safe = torch.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0)
    diff = (pred - safe).pow(2) * m
    return diff.sum() / m.sum().clamp(min=1.0)


def _horizon_weighted_masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    w = weights.view(1, -1)
    m = mask.float() * w
    safe = torch.nan_to_num(torch.where(mask > 0, target, torch.zeros_like(target)))
    return ((pred - safe).pow(2) * m).sum() / m.sum().clamp(min=1.0)


def misnet_loss(
    out: dict[str, torch.Tensor],
    soft_y: torch.Tensor,
    forecast_y: torch.Tensor,
    risk_y: Optional[torch.Tensor] = None,
    soft_y_mask: Optional[torch.Tensor] = None,
    forecast_y_mask: Optional[torch.Tensor] = None,
    weights: MISNetLossWeights | None = None,
    cfg: MISNetConfig | None = None,
) -> dict[str, torch.Tensor]:
    if weights is None:
        weights = MISNetLossWeights()
    if cfg is None:
        cfg = MISNetConfig()

    if soft_y_mask is not None:
        l_soft = masked_mse(out["soft"], soft_y, soft_y_mask)
    else:
        l_soft = F.mse_loss(out["soft"], torch.nan_to_num(soft_y))

    pred = out["forecast"]
    if forecast_y_mask is None:
        forecast_y_mask = torch.ones_like(pred)
    h = pred.size(-1)
    ch_w = torch.tensor(cfg.chla_horizon_weights[:h], device=pred.device, dtype=pred.dtype)
    do_w = torch.tensor(cfg.do_horizon_weights[:h], device=pred.device, dtype=pred.dtype)
    if ch_w.numel() < h:
        pad = torch.ones(h - ch_w.numel(), device=pred.device, dtype=pred.dtype)
        ch_w = torch.cat([ch_w, pad])
        do_w = torch.cat([do_w, pad])
    ch_w = ch_w / ch_w.mean().clamp(min=1e-6)
    do_w = do_w / do_w.mean().clamp(min=1e-6)

    terms = []
    if pred.size(1) >= 1:
        terms.append(
            _horizon_weighted_masked_mse(pred[:, 0], forecast_y[:, 0], forecast_y_mask[:, 0], do_w)
        )
    if pred.size(1) >= 2:
        terms.append(
            _horizon_weighted_masked_mse(pred[:, 1], forecast_y[:, 1], forecast_y_mask[:, 1], ch_w)
        )
    if not terms:
        l_forecast = pred.new_zeros(())
    elif len(terms) == 1:
        l_forecast = terms[0]
    else:
        l_forecast = torch.stack(terms).mean()
        l_forecast = l_forecast + (float(cfg.chla_loss_weight) - 1.0) * terms[1]

    total = (
        weights.soft * l_soft
        + weights.forecast * l_forecast
        + weights.moe_balance * out["moe_loss"]
    )
    losses = {
        "soft": l_soft,
        "forecast": l_forecast,
        "moe_loss": out["moe_loss"],
    }

    if risk_y is not None:
        l_risk = F.binary_cross_entropy_with_logits(out["risk"], risk_y)
        total = total + weights.risk * l_risk
        losses["risk"] = l_risk

    losses["total"] = total
    return losses


def build_misnet_ablation(variant: str, **kwargs) -> MISNet:
    variant = variant.lower().replace("-", "_").replace("w/o_", "wo_")
    ablation_map = {
        "full": {},
        "default": {},
        "wo_missingness_encoder": {"use_missingness_encoder": False},
        "wo_hydro_periodic": {"use_hydro_periodic": False},
        "wo_spectral_mixer": {"use_spectral_mixer": False},
        "wo_sla": {"use_sla": False},
        "wo_mask": {"use_mask": False},
        "no_exo": {"use_exo": False},
        "wo_last_sla_gate": {"use_last_sla_gate": False},
        "wo_persist_residual": {"use_persist_residual": False},
        "wo_mute_dead": {"mute_dead_channels": False},
        "wo_enhance_per_var": {"enhance_per_var": False},
        "lite": {
            "use_hydro_periodic": False,
            "use_spectral_mixer": False,
        },
        "revision_l2": {
            "do_near_persistence": True,
            "do_residual_init": 0.05,
        },
        "revision_l3": {
            "do_near_persistence": True,
            "do_residual_init": 0.05,
            "chla_anchor_mode": "obs_if_valid",
            "chla_loss_weight": 1.5,
            "chla_horizon_weights": (0.9, 0.95, 1.0, 1.05, 1.1, 1.25),
        },
    }
    if variant not in ablation_map:
        raise ValueError(f"Unknown ablation variant: {variant}")
    kwargs.update(ablation_map[variant])
    if kwargs.get("use_exo"):
        kwargs.setdefault("n_exo", 2)
    return build_misnet(**kwargs)


if __name__ == "__main__":
    B, C, L = 4, 5, 168

    def _nparams(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    for name, builder in [
        ("MISNet", lambda: build_misnet(in_channels=C, seq_len=L)),
        ("MISNet-lite", lambda: build_misnet_lite(in_channels=C, seq_len=L)),
        ("MISNet+Exo", lambda: build_misnet_full(in_channels=C, seq_len=L, n_exo=2)),
    ]:
        model = builder()
        x = torch.randn(B, C, L)
        mask = (torch.rand(B, C, L) > 0.1).float()
        exo = torch.randn(B, 2, L) if model.cfg.use_exo else None
        out = model(x, mask, exo)
        print(f"[{name}] params:", _nparams(model))
        print(f"[{name}] soft:", out["soft"].shape)
        print(f"[{name}] forecast:", out["forecast"].shape)
        print(f"[{name}] risk:", out["risk"].shape)
        print(f"[{name}] moe_loss:", out["moe_loss"].item())
        print(f"[{name}] hydro/moe:", model.cfg.use_hydro_periodic, model.cfg.use_spectral_mixer)

        loss = misnet_loss(
            out,
            soft_y=torch.randn(B, 3),
            forecast_y=torch.randn(B, 2, 6),
            soft_y_mask=torch.tensor([[1.0, 0.0, 1.0]] * B),
            forecast_y_mask=torch.ones(B, 2, 6),
            risk_y=torch.randint(0, 2, (B, 3)).float(),
        )
        print(f"[{name}] total loss:", loss["total"].item())


LIMON = MISNet
LIMONConfig = MISNetConfig
LIMONLossWeights = MISNetLossWeights
build_limon = build_misnet
build_limon_forecast_only = build_misnet_forecast_only
build_limon_rev_l2 = build_misnet_rev_l2
build_limon_rev_l3 = build_misnet_rev_l3
build_limon_full = build_misnet_full
build_limon_lite = build_misnet_lite
build_limon_ablation = build_misnet_ablation
limon_loss = misnet_loss
