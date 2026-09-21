from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _xgb_n_jobs() -> int:
    cap = int(os.environ.get("XGB_N_JOBS", "8"))
    cpus = os.cpu_count() or 1
    return max(1, min(cap, cpus))


def _flatten_windows(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        return x.reshape(len(x), -1)
    m = np.nan_to_num(np.asarray(mask, dtype=np.float64), nan=0.0)
    return np.concatenate([x, m], axis=1).reshape(len(x), -1)


def _sanitize_labels(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    invalid = (np.asarray(m) <= 0) | ~np.isfinite(out)
    out[invalid] = 0.0
    return out


@dataclass
class SklearnBundle:
    soft_models: list[Any | None]
    forecast_model: Any | None
    kind: str


class BaselineTCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 64,
        n_soft: int = 3,
        n_forecast: int = 2,
        pred_len: int = 6,
        kernel: int = 3,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_forecast = n_forecast
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel, padding=pad),
            nn.GELU(),
        )
        self.soft_head = nn.Linear(hidden, n_soft)
        self.forecast_head = nn.Linear(hidden, n_forecast * pred_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.net(x)
        ctx = h.mean(dim=-1)
        forecast_flat = self.forecast_head(ctx)
        return {
            "soft": self.soft_head(ctx),
            "forecast": forecast_flat.view(-1, self.n_forecast, self.pred_len),
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
        }


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 64,
        n_soft: int = 3,
        n_forecast: int = 2,
        pred_len: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_forecast = n_forecast
        self.lstm = nn.LSTM(
            in_channels,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.soft_head = nn.Linear(hidden, n_soft)
        self.forecast_head = nn.Linear(hidden, n_forecast * pred_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        ctx = out[:, -1, :]
        forecast_flat = self.forecast_head(ctx)
        return {
            "soft": self.soft_head(ctx),
            "forecast": forecast_flat.view(-1, self.n_forecast, self.pred_len),
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
        }


class BaselineMaskLSTM(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 64,
        n_soft: int = 3,
        n_forecast: int = 2,
        pred_len: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_forecast = n_forecast
        self.lstm = nn.LSTM(
            in_channels * 2,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.soft_head = nn.Linear(hidden, n_soft)
        self.forecast_head = nn.Linear(hidden, n_forecast * pred_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones_like(x)
        else:
            mask = mask.to(dtype=x.dtype)
        x = torch.nan_to_num(x, nan=0.0) * mask
        seq = torch.cat([x, mask], dim=1).transpose(1, 2)
        out, _ = self.lstm(seq)
        ctx = out[:, -1, :]
        forecast_flat = self.forecast_head(ctx)
        return {
            "soft": self.soft_head(ctx),
            "forecast": forecast_flat.view(-1, self.n_forecast, self.pred_len),
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
        }


class BaselineMTLSTM(BaselineLSTM):
    pass


class BaselineGRUD(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 64,
        n_soft: int = 3,
        n_forecast: int = 2,
        pred_len: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.pred_len = pred_len
        self.n_forecast = n_forecast
        self.register_buffer("x_mean", torch.zeros(in_channels), persistent=False)
        self.gamma_x_w = nn.Parameter(torch.full((in_channels,), 0.1))
        self.gamma_x_b = nn.Parameter(torch.zeros(in_channels))
        self.gamma_h_w = nn.Parameter(torch.tensor(0.1))
        self.gamma_h_b = nn.Parameter(torch.tensor(0.0))
        self.cell = nn.GRUCell(in_channels * 2, hidden)
        self.drop = nn.Dropout(dropout)
        self.soft_head = nn.Linear(hidden, n_soft)
        self.forecast_head = nn.Linear(hidden, n_forecast * pred_len)

    @staticmethod
    def _time_deltas(mask: torch.Tensor) -> torch.Tensor:
        b, c, length = mask.shape
        deltas = mask.new_zeros(b, c, length)
        for t in range(1, length):
            deltas[:, :, t] = 1.0 + (1.0 - mask[:, :, t - 1]) * deltas[:, :, t - 1]
        return deltas

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones_like(x)
        else:
            mask = mask.to(dtype=x.dtype)
        b, c, length = x.shape
        deltas = self._time_deltas(mask)
        gx_w = self.gamma_x_w.view(1, -1, 1)
        gx_b = self.gamma_x_b.view(1, -1, 1)
        gamma_x = torch.exp(-torch.relu(gx_w * deltas + gx_b))
        delta_h = deltas.mean(dim=1)
        gamma_h = torch.exp(-torch.relu(self.gamma_h_w * delta_h + self.gamma_h_b))

        x_t = x.permute(2, 0, 1).contiguous()
        m_t = mask.permute(2, 0, 1).contiguous()
        g_t = gamma_x.permute(2, 0, 1).contiguous()
        gh = gamma_h.transpose(0, 1).contiguous()

        h = x.new_zeros(b, self.hidden)
        x_last = self.x_mean.view(1, c).expand(b, c).clone()
        mean = self.x_mean.view(1, c)
        for t in range(length):
            mt = m_t[t]
            xt = x_t[t]
            gt = g_t[t]
            x_imp = mt * xt + (1.0 - mt) * (gt * x_last + (1.0 - gt) * mean)
            x_last = mt * xt + (1.0 - mt) * x_last
            inp = torch.cat([x_imp, mt], dim=-1)
            h = gh[t].unsqueeze(-1) * h
            h = self.cell(inp, h)
        ctx = self.drop(h)
        forecast_flat = self.forecast_head(ctx)
        return {
            "soft": self.soft_head(ctx),
            "forecast": forecast_flat.view(-1, self.n_forecast, self.pred_len),
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
        }


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _ProbAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 4, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        u = min(l, max(1, self.factor * int(math.ceil(math.log(max(l, 2))))))
        if u >= l:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn = self.drop(torch.softmax(scores, dim=-1))
            ctx = torch.matmul(attn, v)
        else:
            sample_k = min(l, u)
            idx_sample = torch.randint(0, l, (sample_k,), device=x.device)
            k_sample = k[:, :, idx_sample]
            scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) / (self.d_k ** 0.5)
            M = scores_sample.max(dim=-1).values - scores_sample.mean(dim=-1)
            top_idx = M.topk(u, dim=-1).indices
            q_top = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
            scores = torch.matmul(q_top, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn = self.drop(torch.softmax(scores, dim=-1))
            ctx_top = torch.matmul(attn, v)
            ctx = v.mean(dim=2, keepdim=True).expand(-1, -1, l, -1).clone()
            ctx.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k), ctx_top)
        ctx = ctx.transpose(1, 2).contiguous().view(b, l, d)
        return self.out(ctx)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float):
        super().__init__()
        self.attn = _ProbAttention(d_model, n_heads, factor, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.n1(x + self.drop(self.attn(x)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class BaselineInformer(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        d_model: int = 64,
        n_soft: int = 3,
        n_forecast: int = 2,
        pred_len: int = 6,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        factor: int = 5,
        max_len: int = 512,
        **kwargs: Any,
    ):
        super().__init__()
        if "hidden" in kwargs and kwargs["hidden"] is not None:
            d_model = int(kwargs["hidden"])
        self.pred_len = pred_len
        self.n_forecast = n_forecast
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [_InformerEncoderLayer(d_model, n_heads, factor, dropout) for _ in range(num_layers)]
        )
        self.distills = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for _ in range(max(0, num_layers - 1))
            ]
        )
        self.soft_head = nn.Linear(d_model, n_soft)
        self.forecast_head = nn.Linear(d_model, n_forecast * pred_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.pos(self.input_proj(x.transpose(1, 2)))
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.distills) and h.size(1) >= 4:
                h = self.distills[i](h.transpose(1, 2)).transpose(1, 2)
        ctx = h.mean(dim=1)
        forecast_flat = self.forecast_head(ctx)
        return {
            "soft": self.soft_head(ctx),
            "forecast": forecast_flat.view(-1, self.n_forecast, self.pred_len),
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
        }


class BaselinePatchTST(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        seq_len: int = 168,
        d_model: int = 64,
        n_soft: int = 3,
        n_forecast: int = 2,
        pred_len: int = 6,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        **kwargs: Any,
    ):
        super().__init__()
        if "hidden" in kwargs and kwargs["hidden"] is not None:
            d_model = int(kwargs["hidden"])
        self.pred_len = pred_len
        self.n_forecast = n_forecast
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.stride = stride
        n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.soft_head = nn.Linear(d_model * in_channels, n_soft)
        self.forecast_head = nn.Linear(d_model * in_channels, n_forecast * pred_len)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(-2) > self.n_patches:
            patches = patches[..., : self.n_patches, :]
        elif patches.size(-2) < self.n_patches:
            pad_n = self.n_patches - patches.size(-2)
            patches = F.pad(patches, (0, 0, 0, pad_n))
        return patches

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, c, _ = x.shape
        patches = self._patchify(x)
        tokens = self.patch_embed(patches)
        tokens = tokens.reshape(b * c, self.n_patches, -1)
        enc = self.encoder(tokens)
        per_ch = self.head(enc)
        ctx = per_ch.view(b, c * per_ch.size(-1))
        forecast_flat = self.forecast_head(ctx)
        return {
            "soft": self.soft_head(ctx),
            "forecast": forecast_flat.view(-1, self.n_forecast, self.pred_len),
            "moe_loss": torch.zeros((), device=x.device, dtype=x.dtype),
        }


TORCH_BASELINE_TYPES = (
    BaselineLSTM,
    BaselineMaskLSTM,
    BaselineTCN,
    BaselineMTLSTM,
    BaselineInformer,
    BaselinePatchTST,
    BaselineGRUD,
)


def baseline_lstm_loss(
    out: dict[str, torch.Tensor],
    soft_y: torch.Tensor,
    forecast_y: torch.Tensor,
    soft_y_mask: torch.Tensor | None = None,
    forecast_y_mask: torch.Tensor | None = None,
    w_soft: float = 1.0,
    w_forecast: float = 1.0,
) -> dict[str, torch.Tensor]:
    from models.limon import masked_mse

    if soft_y_mask is not None:
        l_soft = masked_mse(out["soft"], soft_y, soft_y_mask)
    else:
        l_soft = F.mse_loss(out["soft"], torch.nan_to_num(soft_y))

    if forecast_y_mask is not None:
        l_fc = masked_mse(out["forecast"], forecast_y, forecast_y_mask)
    else:
        l_fc = F.mse_loss(out["forecast"], torch.nan_to_num(forecast_y))

    total = w_soft * l_soft + w_forecast * l_fc
    return {"soft": l_soft, "forecast": l_fc, "total": total, "moe_loss": out["moe_loss"]}


def train_sklearn_baseline(
    model_kind: str,
    x: np.ndarray,
    mask: np.ndarray,
    soft_y: np.ndarray,
    soft_y_mask: np.ndarray,
    forecast_y: np.ndarray,
    forecast_y_mask: np.ndarray,
    tasks: tuple[str, ...],
    seed: int,
    n_components: int = 10,
    max_depth: int = 6,
    xgb_lr: float = 0.05,
    n_estimators: int = 300,
) -> SklearnBundle:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.multioutput import MultiOutputRegressor

    rng = np.random.default_rng(seed)
    features = _flatten_windows(x, mask)
    soft_y = _sanitize_labels(soft_y, soft_y_mask)
    forecast_y = _sanitize_labels(forecast_y, forecast_y_mask)

    soft_models: list[Any | None] = [None, None, None]
    if "soft" in tasks:
        if model_kind == "pls":
            for i in range(soft_y.shape[1]):
                valid = soft_y_mask[:, i] > 0
                if valid.sum() < 5:
                    continue
                n_comp = min(n_components, features[valid].shape[1], int(valid.sum()) - 1)
                m = PLSRegression(n_components=max(1, n_comp))
                m.fit(features[valid], soft_y[valid, i])
                soft_models[i] = m
        elif model_kind == "xgboost":
            import xgboost as xgb

            n_jobs = _xgb_n_jobs()
            for i in range(soft_y.shape[1]):
                valid = soft_y_mask[:, i] > 0
                if valid.sum() < 5:
                    continue
                m = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=xgb_lr,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                    n_jobs=n_jobs,
                    tree_method="hist",
                    verbosity=0,
                )
                m.fit(features[valid], soft_y[valid, i])
                soft_models[i] = m

    forecast_model = None
    if "forecast" in tasks:
        y_flat = forecast_y.reshape(len(forecast_y), -1)
        m_flat = forecast_y_mask.reshape(len(forecast_y_mask), -1)
        valid_rows = m_flat.sum(axis=1) > 0
        if valid_rows.sum() >= 5:
            if model_kind == "pls":
                n_comp = min(n_components, features.shape[1], int(valid_rows.sum()) - 1)
                forecast_model = PLSRegression(n_components=max(1, n_comp))
                forecast_model.fit(features[valid_rows], y_flat[valid_rows])
            elif model_kind == "xgboost":
                import xgboost as xgb

                n_jobs = _xgb_n_jobs()
                forecast_model = MultiOutputRegressor(
                    xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=xgb_lr,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                        n_jobs=n_jobs,
                        tree_method="hist",
                        verbosity=0,
                    ),
                    n_jobs=1,
                )
                forecast_model.fit(features[valid_rows], y_flat[valid_rows])

    return SklearnBundle(soft_models=soft_models, forecast_model=forecast_model, kind=model_kind)


def predict_sklearn_baseline(
    bundle: SklearnBundle,
    x: np.ndarray,
    mask: np.ndarray,
    n_forecast: int,
    pred_len: int,
    n_soft: int = 3,
) -> dict[str, np.ndarray]:
    features = _flatten_windows(x, mask)
    soft = np.zeros((len(x), n_soft), dtype=np.float32)
    for i, m in enumerate(bundle.soft_models):
        if m is not None:
            soft[:, i] = m.predict(features).astype(np.float32)

    forecast = np.zeros((len(x), n_forecast, pred_len), dtype=np.float32)
    if bundle.forecast_model is not None:
        pred_flat = bundle.forecast_model.predict(features).astype(np.float32)
        forecast = pred_flat.reshape(len(x), n_forecast, pred_len)
    return {"soft": soft, "forecast": forecast}
