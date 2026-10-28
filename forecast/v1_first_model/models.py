"""
models.py
=========
Forecasters for the 7-day-ahead Bleaching Alert level of every Marine Park cell.
Each model exposes  predict(cube, t) -> [H, W] array of levels 0-4 (NaN outside the park).

Baselines (no learning, or trivially simple):
  Persistence      - the level in 7 days equals today's level
  Climatology      - the most common level for that cell and time of season in training years
  TrendPhysics     - extrapolate HotSpot and DHW 7 days ahead from last week's trend,
                     then apply NOAA's BAA rule

Learned models:
  GradientBoosting - per-cell tabular model on current values, weekly trends,
                     neighbourhood means and season
  CNNConvLSTM      - the project's CNN + LSTM idea on raw values: a shared CNN encodes each
                     day, a ConvLSTM carries the week forward, a decoder outputs a level per cell
"""

import datetime as dt

import numpy as np
from scipy.ndimage import uniform_filter

from coralwatch import geo
from dataset import HORIZON, SEQ_LEN

N_CLASSES = 5


# ── Baselines ─────────────────────────────────────────────────────────
class Persistence:
    name = "Persistence"

    def fit(self, cube, train_idx, val_idx=None):
        return self

    def predict(self, cube, t):
        return cube.baa[t].copy()


def season_day(day):
    """Days since 1 Nov of the current season (Nov 1 = 0 ... Apr 30 ~ 180)."""
    start = dt.date(day.year if day.month >= 7 else day.year - 1, 11, 1)
    return (day - start).days


class Climatology:
    name = "Climatology"

    def fit(self, cube, train_idx, val_idx=None, window=10):
        days = sorted({cube.dates[t] + dt.timedelta(days=HORIZON) for t in train_idx})
        sd = np.array([season_day(d) for d in days])
        stack = np.stack([cube.baa[cube.index[d]] for d in days])  # [N, H, W]
        self.table = {}
        for s in range(-20, 200):
            sel = np.abs(sd - s) <= window
            if sel.any():
                counts = np.stack([(stack[sel] == k).sum(0) for k in range(N_CLASSES)])
                self.table[s] = counts.argmax(0).astype(float)
        return self

    def predict(self, cube, t):
        target_day = cube.dates[t] + dt.timedelta(days=HORIZON)
        s = min(self.table, key=lambda k: abs(k - season_day(target_day)))
        out = self.table[s].copy()
        out[~cube.valid] = np.nan
        return out


class TrendPhysics:
    name = "Trend + NOAA rule"

    def fit(self, cube, train_idx, val_idx=None):
        return self

    def predict(self, cube, t):
        w = cube.window(t)[:, :3]                # [7, 3, H, W]
        hs_now, dhw_now = w[-1, 1], w[-1, 2]
        hs_trend = (w[-1, 1] - w[0, 1]) / (SEQ_LEN - 1)
        dhw_trend = (w[-1, 2] - w[0, 2]) / (SEQ_LEN - 1)
        hs_future = hs_now + HORIZON * hs_trend
        dhw_future = np.maximum(0, dhw_now + HORIZON * dhw_trend)
        return geo.baa_from(hs_future, dhw_future)


# ── Gradient boosting on per-cell features ────────────────────────────
FEATURES = [
    "ssta", "hotspot", "dhw", "baa",
    "ssta_mean7", "hotspot_mean7", "hotspot_max7",
    "dhw_change3", "dhw_change6", "ssta_change6", "hotspot_change6",
    "hotspot_nbr", "dhw_nbr", "ssta_nbr",
    "season_day", "lat", "lon",
]


def cell_features(cube, t):
    """[n_cells, n_features] for all valid cells on issue day t."""
    w = cube.window(t)[:, :3]                           # [7, 3, H, W]: ssta, hotspot, dhw
    ssta, hs, dhw = w[-1]
    filled = np.nan_to_num(w[-1])
    nbr = [uniform_filter(filled[c], size=5) for c in range(3)]   # ~50 km neighbourhood
    lat2, lon2 = np.meshgrid(cube.lats, cube.lons, indexing="ij")
    cols = [
        ssta, hs, dhw, cube.baa[t],
        np.nanmean(w[:, 0], 0), np.nanmean(w[:, 1], 0), np.nanmax(w[:, 1], 0),
        dhw - w[-4, 2], dhw - w[0, 2], ssta - w[0, 0], hs - w[0, 1],
        nbr[1], nbr[2], nbr[0],
        np.full(ssta.shape, season_day(cube.dates[t])), lat2, lon2,
    ]
    return np.stack([c[cube.valid] for c in cols], axis=1).astype(np.float32)


class GradientBoosting:
    name = "Gradient boosting"

    def __init__(self, max_rows=1_200_000, seed=0):
        self.max_rows, self.seed = max_rows, seed

    def fit(self, cube, train_idx, val_idx=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        rng = np.random.default_rng(self.seed)
        per_day = max(1, self.max_rows // len(train_idx))
        xs, ys = [], []
        for t in train_idx:
            f = cell_features(cube, t)
            y = cube.target(t)[cube.valid]
            keep = rng.choice(len(y), size=min(per_day, len(y)), replace=False)
            xs.append(f[keep]); ys.append(y[keep])
        x, y = np.concatenate(xs), np.concatenate(ys).astype(int)
        self.clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=self.seed,
        )
        self.clf.fit(x, y)
        return self

    def predict(self, cube, t):
        out = np.full(cube.valid.shape, np.nan)
        out[cube.valid] = self.clf.predict(cell_features(cube, t))
        return out

    def predict_proba(self, cube, t):
        out = np.full(cube.valid.shape + (N_CLASSES,), np.nan)
        p = np.zeros((cube.valid.sum(), N_CLASSES))
        p[:, self.clf.classes_.astype(int)] = self.clf.predict_proba(cell_features(cube, t))
        out[cube.valid] = p
        return out


# ── CNN + ConvLSTM ────────────────────────────────────────────────────
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_net(in_ch=6, hidden=32):
    torch, nn = _torch()

    class ConvLSTMCell(nn.Module):
        def __init__(self, ch, hid):
            super().__init__()
            self.hid = hid
            self.gates = nn.Conv2d(ch + hid, 4 * hid, 3, padding=1)

        def forward(self, x, state):
            h, c = state
            i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

    class CNNConvLSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(                       # per-day CNN encoder (shared)
                nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            )
            self.down = nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)
            self.cell = ConvLSTMCell(hidden, hidden)        # temporal memory at half resolution
            self.up = nn.ConvTranspose2d(hidden, hidden, 2, stride=2)
            self.head = nn.Sequential(
                nn.Conv2d(2 * hidden, hidden, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.1),
                nn.Conv2d(hidden, N_CLASSES, 1),
            )

        def forward(self, x):                               # x: [B, T, C, H, W]
            b, t, _, h, w = x.shape
            feats = [self.enc(x[:, k]) for k in range(t)]
            small = [self.down(f) for f in feats]
            hs = torch.zeros_like(small[0]); cs = torch.zeros_like(small[0])
            for s in small:
                hs, cs = self.cell(s, (hs, cs))
            up = self.up(hs)[:, :, :h, :w]
            return self.head(torch.cat([up, feats[-1]], 1))

    return CNNConvLSTMNet()


# Input scaling (physical units -> roughly unit range)
SCALE = np.array([2.0, 2.0, 8.0], dtype=np.float32)


def net_input(cube, t, pad_to):
    """[T, 6, H', W'] : ssta, hotspot, dhw (scaled), valid mask, season sin/cos."""
    w = cube.window(t)[:, :3] / SCALE[None, :, None, None]
    w = np.nan_to_num(w)
    T, _, H, W = w.shape
    sd = season_day(cube.dates[t]) / 181.0 * np.pi
    extra = np.stack([
        np.broadcast_to(cube.valid, (T, H, W)).astype(np.float32),
        np.full((T, H, W), np.sin(sd), np.float32),
        np.full((T, H, W), np.cos(sd), np.float32),
    ], 1)
    x = np.concatenate([w, extra], 1)
    ph, pw = pad_to
    return np.pad(x, ((0, 0), (0, 0), (0, ph - H), (0, pw - W)))


class CNNConvLSTM:
    name = "CNN-ConvLSTM"

    def __init__(self, epochs=40, lr=2e-3, batch=8, patience=8, seed=0):
        self.epochs, self.lr, self.batch, self.patience, self.seed = epochs, lr, batch, patience, seed
        self.history = []

    def _device(self):
        torch, _ = _torch()
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, cube, train_idx, val_idx):
        torch, nn = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        H, W = cube.valid.shape
        self.pad = (H + H % 2, W + W % 2)
        dev = self._device()
        self.net = build_net().to(dev)

        def batch_of(idx):
            x = np.stack([net_input(cube, t, self.pad) for t in idx])
            y = np.full((len(idx),) + self.pad, -1, np.int64)
            for k, t in enumerate(idx):
                tgt = cube.target(t)
                y[k, :H, :W] = np.where(np.isnan(tgt), -1, tgt)
            return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

        # class weights from training targets (sqrt-balanced)
        counts = np.bincount(np.concatenate([cube.target(t)[cube.valid].astype(int) for t in train_idx]), minlength=N_CLASSES)
        weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(dev), ignore_index=-1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        from evaluate import macro_f1  # local import avoids a cycle
        best, best_state, bad = -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            order = np.random.permutation(train_idx)
            total = 0.0
            for s in range(0, len(order), self.batch):
                x, y = batch_of(order[s:s + self.batch])
                opt.zero_grad()
                loss = loss_fn(self.net(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(x)
            sched.step()
            # model selection on the validation seasons only
            preds = [self.predict(cube, t) for t in val_idx]
            score = macro_f1(cube, val_idx, preds)
            self.history.append({"epoch": epoch, "train_loss": total / len(order), "val_macro_f1": score})
            print(f"    epoch {epoch:02d}  loss {total / len(order):.4f}  val macro-F1 {score:.4f}", flush=True)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val = best
        return self

    def predict_proba(self, cube, t):
        torch, _ = _torch()
        H, W = cube.valid.shape
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(net_input(cube, t, self.pad)[None]).to(next(self.net.parameters()).device)
            p = torch.softmax(self.net(x), 1)[0, :, :H, :W].cpu().numpy()
        out = np.moveaxis(p, 0, -1).astype(float)
        out[~cube.valid] = np.nan
        return out

    def predict(self, cube, t):
        p = self.predict_proba(cube, t)
        out = np.nanargmax(np.nan_to_num(p, nan=-1), -1).astype(float)
        out[~cube.valid] = np.nan
        return out

    def save(self, path):
        torch, _ = _torch()
        torch.save({"state": self.net.state_dict(), "pad": self.pad}, path)

    @classmethod
    def load(cls, path):
        torch, _ = _torch()
        obj = cls()
        ckpt = torch.load(path, map_location="cpu")
        obj.net = build_net()
        obj.net.load_state_dict(ckpt["state"])
        obj.pad = tuple(ckpt["pad"])
        return obj
