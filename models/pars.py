from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pars_blocks import (
    DOLinearBypass,
    DirectHorizonHead,
    HorizonQueryHead,
    LabAgeChlaGate,
    MaskAwareScaleMix,
    ObsStateEncoder,
    PeriodQueryChannelMix,
    PersistKoopmanCore,
    SoftCalibHead,
)
from .limon import LIMONLossWeights, masked_mse


@dataclass
class PARSConfig:
    in_channels: int = 5
    seq_len: int = 168
    d_model: int = 64
    n_soft: int = 3
    n_forecast: int = 2
    pred_len: int = 6
    n_queries: int = 6
    period: int = 6
    cross_rank: int = 8
    scales: tuple[int, ...] = (6, 21, 42)
    use_mask: bool = True
    use_obs_decay: bool = True
    use_scale_mix: bool = True
    use_period_mix: bool = True
    use_koopman: bool = True
    use_persist_residual: bool = True
    use_do_linear_bypass: bool = False
    use_avail_attn: bool = False
    use_diurnal: bool = False
    use_stream_gate: bool = False
    use_horizon_query: bool = False
    use_lab_age_gate: bool = False
    mute_dead_channels: bool = True
    dead_channel_thresh: float = 1e-4
    do_channel_index: int = 4
    chla_soft_index: int = 2
    do_lookback: int = 12
    chla_horizon_weights: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
    do_horizon_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    chla_loss_weight: float = 1.0
    soft_aux_weight: float = 1.0
    dropout: float = 0.1


class PARS(nn.Module):
    def __init__(self, cfg: PARSConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = PARSConfig(**kwargs)
        self.cfg = cfg

        if cfg.use_obs_decay:
            self.obs_encoder = ObsStateEncoder(
                cfg.in_channels,
                cfg.d_model,
                mute_dead_channels=cfg.mute_dead_channels,
                dead_channel_thresh=cfg.dead_channel_thresh,
                dropout=cfg.dropout,
                use_diurnal=cfg.use_diurnal,
                diurnal_period=cfg.period,
                use_stream_gate=cfg.use_stream_gate,
            )
        else:
            self.obs_encoder = nn.Sequential(
                nn.Conv1d(cfg.in_channels, cfg.d_model, 3, padding=1),
                nn.GELU(),
                nn.GroupNorm(min(8, cfg.d_model), cfg.d_model),
            )

        self.scale_mix = (
            MaskAwareScaleMix(cfg.d_model, scales=cfg.scales, dropout=cfg.dropout)
            if cfg.use_scale_mix
            else None
        )
        self.period_mix = (
            PeriodQueryChannelMix(
                cfg.d_model,
                n_queries=cfg.n_queries,
                period=cfg.period,
                rank=cfg.cross_rank,
                dropout=cfg.dropout,
            )
            if cfg.use_period_mix
            else None
        )
        self.koopman = (
            PersistKoopmanCore(cfg.d_model, dropout=cfg.dropout) if cfg.use_koopman else None
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.soft_head = SoftCalibHead(cfg.d_model, cfg.n_soft, dropout=cfg.dropout)
        if cfg.use_horizon_query:
            self.forecast_head = HorizonQueryHead(
                cfg.d_model, cfg.n_forecast, cfg.pred_len, dropout=cfg.dropout
            )
        else:
            self.forecast_head = DirectHorizonHead(
                cfg.d_model, cfg.n_forecast, cfg.pred_len, dropout=cfg.dropout
            )
        self.do_bypass = (
            DOLinearBypass(cfg.pred_len, lookback=cfg.do_lookback, dropout=0.0)
            if cfg.use_do_linear_bypass
            else None
        )
        self.lab_age_gate = LabAgeChlaGate() if cfg.use_lab_age_gate else None
        self.readout_gate = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.Sigmoid(),
        )

    def _resolve_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return (~torch.isnan(x)).float()
        return mask

    def encode(
        self, x: torch.Tensor, mask: torch.Tensor, return_diag: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        enc_diag: dict[str, torch.Tensor] = {}
        if cfg.use_mask and cfg.use_obs_decay:
            if return_diag and hasattr(self.obs_encoder, "avail_gate"):
                h, enc_diag = self.obs_encoder(x, mask, return_diag=True)
            else:
                h = self.obs_encoder(x, mask)
        elif cfg.use_obs_decay:
            if return_diag and hasattr(self.obs_encoder, "avail_gate"):
                h, enc_diag = self.obs_encoder(x, torch.ones_like(mask), return_diag=True)
            else:
                h = self.obs_encoder(x, torch.ones_like(mask))
        else:
            alive = (mask.mean(dim=-1, keepdim=True) > cfg.dead_channel_thresh).to(x.dtype)
            h = self.obs_encoder(torch.nan_to_num(x, nan=0.0) * alive)

        if self.scale_mix is not None:
            h = self.scale_mix(h, mask if cfg.use_mask else torch.ones_like(mask))
        period_assign = None
        if self.period_mix is not None:
            avail = mask.mean(dim=1) if cfg.use_avail_attn else None
            if return_diag:
                h, period_assign = self.period_mix(h, avail=avail, return_assign=True)
            else:
                h = self.period_mix(h, avail=avail)
        if self.koopman is not None:
            h = self.koopman(h)
        if return_diag:
            if period_assign is not None:
                enc_diag["period_assign"] = period_assign
            return h, enc_diag
        return h

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        exo: Optional[torch.Tensor] = None,
        soft_y_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        del exo
        cfg = self.cfg
        mask = self._resolve_mask(x, mask)
        enc, enc_diag = self.encode(x, mask, return_diag=True)
        last = enc[:, :, -1]
        pooled = self.pool(enc).squeeze(-1)
        g = self.readout_gate(torch.cat([last, pooled], dim=-1))
        readout_local_frac = g.mean(dim=-1)
        ctx = g * last + (1.0 - g) * pooled

        soft = self.soft_head(ctx)
        if cfg.use_horizon_query:
            residual = self.forecast_head(enc, ctx)
        else:
            residual = self.forecast_head(ctx)

        do_i = min(max(cfg.do_channel_index, 0), x.size(1) - 1)
        do_last = torch.nan_to_num(x[:, do_i, -1], nan=0.0) * mask[:, do_i, -1]
        chla_i = min(max(cfg.chla_soft_index, 0), soft.size(1) - 1)
        chla_t0 = soft[:, chla_i].detach()

        do_delta = residual[:, 0, :]
        ch_delta = residual[:, 1, :] if cfg.n_forecast > 1 else None
        lab_scale = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        if cfg.use_persist_residual:
            if self.do_bypass is not None:
                do_delta = self.do_bypass(x[:, do_i, :], mask[:, do_i, :])
            if ch_delta is not None and self.lab_age_gate is not None:
                lab_scale = self.lab_age_gate(mask, soft_y_mask, cfg.chla_soft_index)
                ch_delta = ch_delta * lab_scale
            parts = [do_last.unsqueeze(-1) + do_delta]
            if ch_delta is not None:
                parts.append(chla_t0.unsqueeze(-1) + ch_delta)
            forecast = torch.stack(parts, dim=1)
        else:
            forecast = residual

        eps = 1e-4
        do_anchor_h = do_last.unsqueeze(-1).expand_as(do_delta)
        do_depart = do_delta.abs() / (do_anchor_h.abs() + do_delta.abs() + eps)
        diagnostics: dict[str, torch.Tensor] = {
            "channel_avail": mask.mean(dim=-1),
            "readout_local_frac": readout_local_frac,
            "do_anchor": do_last,
            "do_delta": do_delta,
            "do_depart_frac": do_depart,
            "lab_age_scale": lab_scale.squeeze(-1),
        }
        if "channel_gate" in enc_diag and enc_diag["channel_gate"] is not None:
            diagnostics["channel_gate"] = enc_diag["channel_gate"]
        if enc_diag.get("stream_gate") is not None:
            diagnostics["stream_gate"] = enc_diag["stream_gate"].mean(dim=-1)
        if "period_assign" in enc_diag:
            pa = enc_diag["period_assign"]
            diagnostics["period_query_mass"] = pa.mean(dim=1)
            diagnostics["period_query_mass_recent"] = pa[:, -12:, :].mean(dim=1)
        if ch_delta is not None:
            ch_anchor_h = chla_t0.unsqueeze(-1).expand_as(ch_delta)
            diagnostics["chla_anchor"] = chla_t0
            diagnostics["chla_delta"] = ch_delta
            diagnostics["chla_depart_frac"] = ch_delta.abs() / (
                ch_anchor_h.abs() + ch_delta.abs() + eps
            )

        return {
            "soft": soft,
            "forecast": forecast,
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
            "hidden": enc,
            "diagnostics": diagnostics,
        }


def build_pars(**kwargs) -> PARS:
    return PARS(PARSConfig(**kwargs))


def build_pars_forecast_only(**kwargs) -> PARS:
    return build_pars(**kwargs)


def build_pars_ablation(variant: str, **kwargs) -> PARS:
    variant = variant.lower().replace("-", "_").replace("w/o_", "wo_")
    ablation_map = {
        "full": {},
        "default": {},
        "wo_obs_decay": {"use_obs_decay": False},
        "wo_scale_mix": {"use_scale_mix": False},
        "wo_period_mix": {"use_period_mix": False},
        "wo_koopman": {"use_koopman": False},
        "wo_persist_residual": {"use_persist_residual": False},
        "wo_do_linear_bypass": {"use_do_linear_bypass": False},
        "wo_avail_attn": {"use_avail_attn": False},
        "wo_diurnal": {"use_diurnal": False},
        "wo_stream_gate": {"use_stream_gate": False},
        "wo_horizon_query": {"use_horizon_query": False},
        "wo_lab_age_gate": {"use_lab_age_gate": False},
        "wo_mask": {"use_mask": False},
        "ionet_mild": {
            "use_diurnal": True,
            "use_stream_gate": True,
            "chla_loss_weight": 1.5,
            "use_horizon_query": False,
            "use_lab_age_gate": False,
        },
        "ionet_full": {
            "use_diurnal": True,
            "use_stream_gate": True,
            "use_horizon_query": True,
            "use_lab_age_gate": True,
            "chla_loss_weight": 1.5,
        },
    }
    if variant not in ablation_map:
        raise ValueError(f"Unknown PARS ablation: {variant}")
    kwargs.update(ablation_map[variant])
    return build_pars(**kwargs)


def _horizon_weighted_masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    w = weights.view(1, -1)
    m = mask.float() * w
    safe = torch.nan_to_num(torch.where(mask > 0, target, torch.zeros_like(target)))
    return ((pred - safe).pow(2) * m).sum() / m.sum().clamp(min=1.0)


def pars_loss(
    out: dict[str, torch.Tensor],
    soft_y: torch.Tensor,
    forecast_y: torch.Tensor,
    risk_y: Optional[torch.Tensor] = None,
    soft_y_mask: Optional[torch.Tensor] = None,
    forecast_y_mask: Optional[torch.Tensor] = None,
    weights: LIMONLossWeights | None = None,
    cfg: PARSConfig | None = None,
) -> dict[str, torch.Tensor]:
    del risk_y
    if weights is None:
        weights = LIMONLossWeights()
    if cfg is None:
        cfg = PARSConfig()

    soft_w = weights.soft * float(cfg.soft_aux_weight)
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

    total = soft_w * l_soft + weights.forecast * l_forecast + weights.moe_balance * out["moe_loss"]
    return {"soft": l_soft, "forecast": l_forecast, "moe_loss": out["moe_loss"], "total": total}


def apply_structured_outage_batch(
    x: torch.Tensor,
    mask: torch.Tensor,
    prob: float = 0.15,
    min_span: int = 12,
    max_span: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prob <= 0:
        return x, mask
    b, c, length = x.shape
    x = x.clone()
    mask = mask.clone()
    min_span = max(1, min(int(min_span), length))
    max_span = max(min_span, min(int(max_span), length))
    for i in range(b):
        for ch in range(c):
            if float(mask[i, ch].mean()) < 1e-6:
                continue
            if torch.rand((), device=x.device).item() > prob:
                continue
            span = int(torch.randint(min_span, max_span + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, max(1, length - span + 1), (1,), device=x.device).item())
            mask[i, ch, start : start + span] = 0
            x[i, ch, start : start + span] = 0
    return x, mask




LakeForge = PARS
LakeForgeConfig = PARSConfig
build_lake_forge = build_pars
build_lake_forge_forecast_only = build_pars_forecast_only
build_lake_forge_ablation = build_pars_ablation
lake_forge_loss = pars_loss

if __name__ == "__main__":
    m = build_pars()
    x = torch.randn(2, 5, 168)
    mask = (torch.rand(2, 5, 168) > 0.3).float()
    o = m(x, mask)
    print(sum(p.numel() for p in m.parameters()), o["forecast"].shape)
