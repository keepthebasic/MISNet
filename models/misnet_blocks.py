from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F




class LRMaskFormer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel: int = 5,
        mute_dead_channels: bool = True,
        dead_channel_thresh: float = 1e-4,
        enhance_per_var: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mute_dead_channels = mute_dead_channels
        self.dead_channel_thresh = dead_channel_thresh
        self.enhance_per_var = enhance_per_var
        per_var = max(16 if enhance_per_var else 8, d_model // max(in_channels, 1))
        self.per_var = per_var
        pad = kernel // 2
        if enhance_per_var:
            self.var_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(2, per_var, kernel, padding=pad),
                        nn.GELU(),
                        nn.Conv1d(per_var, per_var, kernel, padding=pad, groups=per_var),
                        nn.GELU(),
                        nn.Conv1d(per_var, per_var, 1),
                    )
                    for _ in range(in_channels)
                ]
            )
            self.var_skip = nn.ModuleList(
                [nn.Conv1d(2, per_var, 1) for _ in range(in_channels)]
            )
        else:
            self.var_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(2, per_var, kernel, padding=pad),
                        nn.GELU(),
                        nn.Conv1d(per_var, per_var, 1),
                    )
                    for _ in range(in_channels)
                ]
            )
            self.var_skip = None
        self.var_type = nn.Embedding(in_channels, per_var)
        self.avail_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(per_var * in_channels, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
        )
        self.norm = nn.GroupNorm(min(8, d_model), d_model)

    def channel_alive(self, mask: torch.Tensor) -> torch.Tensor:
        avail = mask.mean(dim=-1)
        if not self.mute_dead_channels:
            return torch.ones_like(avail)
        return (avail > self.dead_channel_thresh).to(dtype=avail.dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = (~torch.isnan(x)).float()
        x = torch.nan_to_num(x, nan=0.0)
        alive = self.channel_alive(mask)
        x = x * alive.unsqueeze(-1)
        mask = mask * alive.unsqueeze(-1)

        branches: list[torch.Tensor] = []
        for c, conv in enumerate(self.var_convs):
            xc = torch.cat([x[:, c : c + 1], mask[:, c : c + 1]], dim=1)
            h = conv(xc)
            if self.var_skip is not None:
                h = h + self.var_skip[c](xc)
            vt = self.var_type.weight[c].view(1, -1, 1).expand_as(h)
            branches.append(h + vt)

        avail = mask.mean(dim=-1)
        w = self.avail_gate(avail) * alive
        weighted: list[torch.Tensor] = []
        for c in range(self.in_channels):
            chunk = branches[c]
            weighted.append(chunk * w[:, c : c + 1].unsqueeze(-1))
        stacked = torch.cat(weighted, dim=1)
        out = self.fuse(stacked)
        return self.norm(out)




class _LearnableBandPass(nn.Module):

    def __init__(self, channels: int, n_bands: int):
        super().__init__()
        self.n_bands = n_bands
        self.centers = nn.Parameter(torch.linspace(0.05, 0.45, n_bands))
        self.log_width = nn.Parameter(torch.zeros(n_bands))
        self.proj = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1) for _ in range(n_bands)]
        )

    def _masks(self, freq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(freq_len, device=device, dtype=torch.float32)
        norm = idx / max(freq_len - 1, 1)
        centers = torch.sigmoid(self.centers)
        width = F.softplus(self.log_width) * 0.08 + 0.02
        diff = (norm.unsqueeze(0) - centers.unsqueeze(1)) / width.unsqueeze(1)
        return torch.exp(-0.5 * diff.pow(2))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        length = x.size(-1)
        xf = torch.fft.rfft(x, dim=-1)
        freq_len = xf.size(-1)
        masks = self._masks(freq_len, x.device)
        bands: list[torch.Tensor] = []
        for i in range(self.n_bands):
            masked = xf * masks[i].view(1, 1, -1)
            xt = torch.fft.irfft(masked, n=length, dim=-1)
            bands.append(self.proj[i](xt))
        return bands


class _UNetLite(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.bot = nn.Conv1d(channels, channels, 3, padding=1)
        self.dec2 = nn.Conv1d(channels * 2, channels, 1)
        self.dec1 = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x) + x
        d1 = F.avg_pool1d(s1, 2)
        s2 = self.enc2(d1) + d1
        d2 = F.avg_pool1d(s2, 2)
        b = self.bot(d2) + d2
        u2 = F.interpolate(b, size=s2.size(-1), mode="linear", align_corners=False)
        u2 = self.dec2(torch.cat([u2, s2], dim=1)) + s2
        u1 = F.interpolate(u2, size=s1.size(-1), mode="linear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, s1], dim=1)) + s1
        return u1


class HyCoPeriodNet(nn.Module):

    def __init__(self, channels: int, n_period_bands: int = 3):
        super().__init__()
        self.band_pass = _LearnableBandPass(channels, n_period_bands)
        self.trend_mix = nn.Parameter(torch.ones(n_period_bands) / n_period_bands)
        self.residual_net = _UNetLite(channels)
        self.out_proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.band_pass(x)
        alpha = F.softmax(self.trend_mix, dim=0)
        trend = sum(a * b for a, b in zip(alpha, bands))
        residual = x - trend
        encoded = self.residual_net(residual)
        return self.out_proj(trend + encoded)




class _LinearTemporalMixer(nn.Module):

    def __init__(self, channels: int, kernel: int = 15):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(channels, channels, kernel, padding=pad, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.gate = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        g = torch.sigmoid(self.gate(x))
        return x + self.pw(h * g)


class CoupledSpecMoE(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_bands: int = 3,
        rank: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.boundaries = nn.Parameter(torch.linspace(0.2, 0.7, max(n_bands - 1, 1)))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, 3, padding=1),
                    nn.GELU(),
                    _LinearTemporalMixer(d_model),
                )
                for _ in range(n_bands)
            ]
        )
        self.task_sync = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.task_forecast = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.sync_gate = nn.Linear(d_model * 2, n_bands)
        self.forecast_gate = nn.Linear(d_model * 2, n_bands)
        self.low_u = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.low_v = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _band_split(self, x: torch.Tensor) -> list[torch.Tensor]:
        xf = torch.fft.rfft(x, dim=-1)
        freq_len = xf.size(-1)
        length = x.size(-1)
        b = torch.sigmoid(self.boundaries)
        b = torch.sort(b)[0]
        edges = torch.cat(
            [torch.zeros(1, device=x.device), b, torch.ones(1, device=x.device)]
        )
        bands: list[torch.Tensor] = []
        for i in range(self.n_bands):
            start = int(edges[i].item() * freq_len)
            end = max(start + 1, int(edges[i + 1].item() * freq_len))
            m = torch.zeros(freq_len, device=x.device)
            m[start:end] = 1.0
            xt = torch.fft.irfft(xf * m.view(1, 1, -1), n=length, dim=-1)
            bands.append(xt)
        return bands

    def _low_rank_refine(self, h: torch.Tensor) -> torch.Tensor:
        b, d, length = h.shape
        flat = h.transpose(1, 2).reshape(b * length, d)
        refined = flat + (flat @ self.low_u) @ self.low_v.t()
        return refined.view(b, length, d).transpose(1, 2)

    @staticmethod
    def _mix(
        weights: torch.Tensor, expert_outs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(expert_outs, dim=1)
        w = weights.softmax(dim=-1).unsqueeze(-1).unsqueeze(-1)
        mixed = (stacked * w).sum(dim=1)
        pools = [e.mean(dim=-1) for e in expert_outs]
        pool_stack = torch.stack(pools, dim=1)
        w_pool = weights.softmax(dim=-1).unsqueeze(-1)
        ctx = (pool_stack * w_pool).sum(dim=1)
        return mixed, ctx

    @staticmethod
    def _balance_loss(gate_logits: torch.Tensor, n_bands: int) -> torch.Tensor:
        usage = gate_logits.softmax(-1).mean(0)
        return (usage * n_bands - 1.0).pow(2).mean()

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bands = self._band_split(h)
        expert_outs = [exp(b) for exp, b in zip(self.experts, bands)]
        pool = h.mean(dim=-1)
        sync_logits = self.sync_gate(
            torch.cat([pool, self.task_sync.expand(pool.size(0), -1)], dim=-1)
        )
        fc_logits = self.forecast_gate(
            torch.cat([pool, self.task_forecast.expand(pool.size(0), -1)], dim=-1)
        )
        h_sync, sync_feat = self._mix(sync_logits, expert_outs)
        h_fc, forecast_feat = self._mix(fc_logits, expert_outs)
        h_out = self._low_rank_refine(h_sync + h_fc)
        h_out = self.dropout(h_out)
        z = h_out.transpose(1, 2)
        z = self.norm(z + z.mean(dim=1, keepdim=True))
        balance = self._balance_loss(sync_logits, self.n_bands) + self._balance_loss(
            fc_logits, self.n_bands
        )
        return z.transpose(1, 2), sync_feat, forecast_feat, balance




class SLAReadout(nn.Module):

    def __init__(
        self,
        d_model: int,
        in_channels: int,
        mute_dead_channels: bool = True,
        dead_channel_thresh: float = 1e-4,
    ):
        super().__init__()
        self.mute_dead_channels = mute_dead_channels
        self.dead_channel_thresh = dead_channel_thresh
        self.score = nn.Conv1d(d_model, 1, 1)
        self.lab_proxy = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        avail = mask.mean(dim=-1)
        if self.mute_dead_channels:
            avail = avail * (avail > self.dead_channel_thresh).to(dtype=avail.dtype)
        bias = self.lab_proxy(avail).unsqueeze(-1)
        logits = self.score(h) + bias
        w = torch.softmax(logits, dim=-1)
        return (h * w).sum(dim=-1)


class ExogenousGate(nn.Module):

    def __init__(self, d_model: int, n_exo: int):
        super().__init__()
        self.proj = nn.Conv1d(n_exo, d_model, 1)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, exo: torch.Tensor | None) -> torch.Tensor:
        if exo is None:
            return feat
        exo_ctx = self.proj(exo).mean(dim=-1)
        g = self.gate(torch.cat([feat, exo_ctx], dim=-1))
        return feat * g + exo_ctx * (1 - g)
