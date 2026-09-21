from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_since_obs(mask: torch.Tensor) -> torch.Tensor:
    b, c, length = mask.shape
    deltas = mask.new_zeros(b, c, length)
    for t in range(1, length):
        deltas[:, :, t] = 1.0 + (1.0 - mask[:, :, t - 1]) * deltas[:, :, t - 1]
    return deltas


def normalize_delta(delta: torch.Tensor, seq_len: int) -> torch.Tensor:
    denom = max(float(seq_len - 1), 1.0)
    return torch.log1p(delta) / torch.log1p(
        torch.tensor(denom, device=delta.device, dtype=delta.dtype)
    )


class DiurnalDetrend(nn.Module):

    def __init__(self, in_channels: int, period: int = 6):
        super().__init__()
        self.period = period
        self.template = nn.Parameter(torch.zeros(in_channels, period))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        phase = torch.arange(length, device=x.device) % self.period
        return x - self.template[:, phase].unsqueeze(0)


class ObsStateEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel: int = 5,
        mute_dead_channels: bool = True,
        dead_channel_thresh: float = 1e-4,
        dropout: float = 0.1,
        use_diurnal: bool = False,
        diurnal_period: int = 6,
        use_stream_gate: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mute_dead_channels = mute_dead_channels
        self.dead_channel_thresh = dead_channel_thresh
        self.diurnal = DiurnalDetrend(in_channels, diurnal_period) if use_diurnal else None
        self.use_stream_gate = use_stream_gate
        per_var = max(12, d_model // max(in_channels, 1))
        pad = kernel // 2
        self.gamma_w = nn.Parameter(torch.full((in_channels,), 0.1))
        self.gamma_b = nn.Parameter(torch.zeros(in_channels))
        self.var_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(3, per_var, kernel, padding=pad),
                    nn.GELU(),
                    nn.Conv1d(per_var, per_var, kernel, padding=pad, groups=per_var),
                    nn.GELU(),
                    nn.Conv1d(per_var, per_var, 1),
                )
                for _ in range(in_channels)
            ]
        )
        self.var_skip = nn.ModuleList([nn.Conv1d(3, per_var, 1) for _ in range(in_channels)])
        self.var_type = nn.Embedding(in_channels, per_var)
        self.avail_gate = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv1d(per_var * in_channels, d_model, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 3, padding=1),
        )
        self.norm = nn.GroupNorm(min(8, d_model), d_model)
        self.stream_gate = (
            nn.Sequential(nn.Linear(in_channels, d_model), nn.Sigmoid())
            if use_stream_gate
            else None
        )

    def channel_alive(self, mask: torch.Tensor) -> torch.Tensor:
        avail = mask.mean(dim=-1)
        if not self.mute_dead_channels:
            return torch.ones_like(avail)
        return (avail > self.dead_channel_thresh).to(dtype=avail.dtype)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, return_diag: bool = False
    ):
        if mask is None:
            mask = (~torch.isnan(x)).float()
        x = torch.nan_to_num(x, nan=0.0)
        alive = self.channel_alive(mask)
        x = x * alive.unsqueeze(-1)
        mask = mask * alive.unsqueeze(-1)
        if self.diurnal is not None:
            x = self.diurnal(x) * mask

        deltas = time_since_obs(mask)
        gx = torch.exp(-F.relu(self.gamma_w.view(1, -1, 1) * deltas + self.gamma_b.view(1, -1, 1)))

        branches = []
        for c, conv in enumerate(self.var_convs):
            xc = torch.cat(
                [x[:, c : c + 1], mask[:, c : c + 1], gx[:, c : c + 1]],
                dim=1,
            )
            h = conv(xc) + self.var_skip[c](xc)
            vt = self.var_type.weight[c].view(1, -1, 1).expand_as(h)
            branches.append(h + vt)
        avail = mask.mean(dim=-1)
        w = self.avail_gate(avail) * alive
        weighted = [branches[c] * w[:, c : c + 1].unsqueeze(-1) for c in range(self.in_channels)]
        out = self.norm(self.fuse(torch.cat(weighted, dim=1)))
        stream = None
        if self.stream_gate is not None:
            stream = self.stream_gate(avail)
            out = out * stream.unsqueeze(-1)
        if return_diag:
            return out, {"channel_gate": w, "stream_gate": stream, "alive": alive}
        return out


class MaskAwareScaleMix(nn.Module):
    def __init__(self, channels: int, scales: tuple[int, ...] = (6, 21, 42), dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        n = 1 + len(scales)
        self.mix = nn.Sequential(
            nn.Conv1d(channels * n, channels, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.proj_mask = nn.Conv1d(1, 1, 1, bias=False)
        nn.init.constant_(self.proj_mask.weight, 1.0)

    def _mask_weight(self, mask: torch.Tensor) -> torch.Tensor:
        w = mask.mean(dim=1, keepdim=True).clamp(min=0.0)
        return self.proj_mask(w).clamp(min=1e-4)

    def _masked_pool(self, h: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
        length = h.size(-1)
        k = min(max(int(k), 1), length)
        num = F.avg_pool1d(h * w, kernel_size=k, stride=k, ceil_mode=True)
        den = F.avg_pool1d(w.expand_as(h), kernel_size=k, stride=k, ceil_mode=True).clamp(min=1e-4)
        return F.interpolate(num / den, size=length, mode="linear", align_corners=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = self._mask_weight(mask)
        bands = [h] + [self._masked_pool(h, w, k) for k in self.scales]
        return self.norm(h + self.mix(torch.cat(bands, dim=1)))


class PeriodQueryChannelMix(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_queries: int = 6,
        period: int = 6,
        rank: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.period = period
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.u = nn.Linear(d_model, rank, bias=False)
        self.v = nn.Linear(rank, d_model, bias=False)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.time_to_query = nn.Linear(d_model, n_queries)

    def forward(self, h: torch.Tensor, avail: torch.Tensor | None = None, return_assign: bool = False):
        x = h.transpose(1, 2)
        b, length, d = x.shape
        q = self.q_proj(self.queries).unsqueeze(0).expand(b, -1, -1)
        k = self.k_proj(x)
        v = self.v_proj(x)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (d**-0.5)
        if avail is not None:
            a = avail.clamp(0.0, 1.0).unsqueeze(1)
            logits = logits + torch.log(a.clamp(min=1e-3))
        attn = torch.softmax(logits, dim=-1)
        ctx = torch.matmul(attn, v)
        assign = torch.softmax(self.time_to_query(x), dim=-1)
        mixed = self.out_proj(self.drop(torch.matmul(assign, ctx)))
        x = self.norm1(x + mixed)
        lr = self.v(self.u(x))
        g = self.gate(torch.cat([x, lr], dim=-1))
        x = x + g * lr
        x = self.norm2(x + self.drop(self.ff(x)))
        out = x.transpose(1, 2)
        if return_assign:
            return out, assign
        return out


class PersistKoopmanCore(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.A = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.A.weight)
        self.ma = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.res = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.norm = nn.GroupNorm(min(8, d_model), d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.transpose(1, 2)
        lin = (flat + self.A(flat)).transpose(1, 2)
        return self.norm(lin + self.ma(h) + self.res(h))


class SoftCalibHead(nn.Module):
    def __init__(self, d_model: int, n_soft: int, dropout: float = 0.1):
        super().__init__()
        self.prior = nn.Parameter(torch.zeros(n_soft))
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_soft),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.prior.view(1, -1) + self.net(ctx)


class DirectHorizonHead(nn.Module):

    def __init__(self, d_model: int, n_forecast: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.n_forecast = n_forecast
        self.pred_len = pred_len
        self.horizon_emb = nn.Embedding(pred_len, d_model)
        self.base = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(d_model, n_forecast)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        steps = []
        for t in range(self.pred_len):
            he = self.horizon_emb.weight[t]
            steps.append(self.delta(self.base(ctx + he.unsqueeze(0))))
        return torch.stack(steps, dim=-1)


class DOLinearBypass(nn.Module):

    def __init__(self, pred_len: int = 6, lookback: int = 12, dropout: float = 0.0):
        super().__init__()
        self.pred_len = pred_len
        self.lookback = lookback
        self.ma = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        nn.init.constant_(self.ma.weight, 1.0 / 5.0)
        self.lin = nn.Linear(lookback, pred_len)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.scale = nn.Parameter(torch.tensor(0.05))
        self.drop = nn.Dropout(dropout)

    def forward(self, do_series: torch.Tensor, do_mask: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(do_series, nan=0.0) * do_mask
        ma = self.ma(x.unsqueeze(1)).squeeze(1)
        last = x[:, -1]
        ma_last = ma[:, -1]
        lb = self.lookback
        if x.size(-1) >= lb:
            window = x[:, -lb:]
            wmask = do_mask[:, -lb:]
        else:
            pad = lb - x.size(-1)
            window = F.pad(x, (pad, 0))
            wmask = F.pad(do_mask, (pad, 0))
        window = window * wmask
        denom = wmask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        filled = window + (1.0 - wmask) * (window.sum(dim=-1, keepdim=True) / denom)
        delta = self.lin(self.drop(filled))
        tip = (ma_last - last).unsqueeze(-1).expand_as(delta)
        return (delta + 0.1 * tip) * self.scale


class HorizonQueryHead(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_forecast: int,
        pred_len: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_forecast = n_forecast
        self.pred_len = pred_len
        self.queries = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_forecast),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, enc: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        keys = enc.transpose(1, 2)
        b = keys.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1) + 0.25 * ctx.unsqueeze(1)
        horizon_ctx, _ = self.attn(q, keys, keys, need_weights=False)
        horizon_ctx = self.norm(horizon_ctx + q)
        return self.delta(horizon_ctx).permute(0, 2, 1)


class LabAgeChlaGate(nn.Module):

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def forward(
        self,
        mask: torch.Tensor,
        soft_y_mask: torch.Tensor | None,
        chla_soft_index: int,
    ) -> torch.Tensor:
        recent = mask[:, :, -12:].mean(dim=-1).mean(dim=-1, keepdim=True)
        end = mask[:, :, -1].mean(dim=-1, keepdim=True)
        if soft_y_mask is not None:
            i = min(max(chla_soft_index, 0), soft_y_mask.size(1) - 1)
            lab = soft_y_mask[:, i : i + 1]
        else:
            lab = torch.ones_like(recent)
        feat = torch.cat([recent, end, lab], dim=-1)
        return torch.sigmoid(self.mlp(feat))
