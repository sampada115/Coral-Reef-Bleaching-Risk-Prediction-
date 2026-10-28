"""
extract_embeddings.py
=====================
Daily CNN-ConvLSTM embeddings from the trained physics-guided model (PhysNet), for
experiments such as fusion with the NLP embeddings. Same layout as the old
CNN/5_results/Embeddings/cnn_embeddings_daily.csv, so it can be swapped in directly.

    .venv/bin/python forecast/7_embeddings/extract_embeddings.py              # model max7_s0
    .venv/bin/python forecast/7_embeddings/extract_embeddings.py --model max7_s1
    .venv/bin/python forecast/7_embeddings/extract_embeddings.py --cells      # also per-cell (~700 MB)

Where the embedding comes from
    For each day, the model's encoder reads the last 4 weeks of maps (7 daily + 3 weekly
    frames: SST anomaly, HotSpot, DHW, SST, past weather, ENSO/MJO, position, season).
    The embedding of a reef cell is [ConvLSTM memory (32) | encoder output for today (32)]
    = 64 numbers. This is taken BEFORE the layers that see the future weather and the
    known 7-day window, so it contains only information available on that day (no leakage).

Daily vector (256 numbers) = 4 blocks of 64, pooled over the Marine Park cells:
    emb_0..63     mean over all park cells
    emb_64..127   max over all park cells (picks up hot spots in small areas)
    emb_128..191  mean over the Northern GBR cells
    emb_192..255  mean over the Central GBR cells
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FORECAST_DIR = HERE.parent
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from coralwatch import geo  # noqa: E402
from dataset import SPLITS, Cube, is_scored, season_of  # noqa: E402
from physnet import Drivers, PhysNet  # noqa: E402
from paths import MODELS_V2, WEATHER  # noqa: E402

LABELS_CSV = FORECAST_DIR.parent / "CNN" / "baa_labels.csv"
DIM = 64


def split_of(day):
    s = season_of(day)
    return next((name for name, seasons in SPLITS.items() if s in seasons), "none")


def full_history(cube, day):
    """True if all 28 days of input history exist (otherwise missing days were zero-filled)."""
    return all(day - dt.timedelta(days=k) in cube.index for k in range(28))


def embed_batch(model, idx):
    """Per-cell embeddings [B, 64, H, W] for issue indices idx, from past/present data only."""
    import torch
    b, net = model.builder, model.net
    fr = torch.from_numpy(np.stack([b.pad2(b.frames(t)) for t in idx])).to(model.dev)
    h, w = b.c.valid.shape
    with torch.no_grad():
        feats = [net.enc(fr[:, k]) for k in range(fr.shape[1])]
        hs = cs = None
        for f in feats:
            s = net.down(f)
            if hs is None:
                hs, cs = torch.zeros_like(s), torch.zeros_like(s)
            hs, cs = net.cell(s, hs, cs)
        e = torch.cat([net.up(hs)[:, :, :h, :w], feats[-1][:, :, :h, :w]], 1)
    return e.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="max7_s0", help="saved model in forecast/6_saved_models/v2/")
    ap.add_argument("--cells", action="store_true", help="also save per-cell embeddings [days, cells, 64] (float16)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    cube = Cube()
    drivers = Drivers(WEATHER, cube.lats, cube.lons)
    model = PhysNet.load(MODELS_V2 / f"{args.model}.pt", cube, drivers)
    model.net.eval()

    valid = cube.valid
    rows = geo.zone_rows(cube.lats)
    zone = {z: valid & rows[z][:, None] for z in ("N", "C")}
    old_labels = {}
    if LABELS_CSV.exists():
        for line in LABELS_CSV.read_text().splitlines()[1:]:
            d, v = line.split(",")
            old_labels[d] = v

    T = len(cube.dates)
    daily = np.zeros((T, 4 * DIM), np.float32)
    cells = np.zeros((T, int(valid.sum()), DIM), np.float16) if args.cells else None
    for s in range(0, T, args.batch):
        idx = list(range(s, min(T, s + args.batch)))
        e = embed_batch(model, idx)                                   # [B, 64, H, W]
        for k, t in enumerate(idx):
            park = e[k][:, valid]                                     # [64, N]
            daily[t] = np.concatenate([park.mean(1), park.max(1),
                                       e[k][:, zone["N"]].mean(1), e[k][:, zone["C"]].mean(1)])
            if cells is not None:
                cells[t] = park.T.astype(np.float16)
        if s % (args.batch * 40) == 0:
            print(f"  {cube.dates[idx[-1]]}  ({idx[-1] + 1}/{T})", flush=True)

    baa7 = cube.baa7
    meta_cols = ["date", "split", "warm_season", "full_history", "baa_label",
                 "alert_level_max", "alert_share", "dhw_mean"]
    lines = [",".join(meta_cols + [f"emb_{i}" for i in range(4 * DIM)])]
    for t, day in enumerate(cube.dates):
        b7 = baa7[t][valid]
        dhw = cube.x[t, 2][valid]
        meta = [day.isoformat(), split_of(day), int(is_scored(day)), int(full_history(cube, day)),
                old_labels.get(day.isoformat(), ""),
                "" if np.isnan(b7).all() else int(np.nanmax(b7)),
                "" if np.isnan(b7).all() else f"{np.nanmean(b7 >= 3):.4f}",
                f"{np.nanmean(dhw):.3f}"]
        lines.append(",".join(map(str, meta)) + "," + ",".join(f"{v:.5f}" for v in daily[t]))

    tag = "" if args.model == "max7_s0" else f"_{args.model}"
    (HERE / f"physnet_embeddings_daily{tag}.csv").write_text("\n".join(lines) + "\n")
    np.save(HERE / f"physnet_embeddings_daily{tag}.npy", daily)
    if cells is not None:
        lat2, lon2 = np.meshgrid(cube.lats, cube.lons, indexing="ij")
        np.savez_compressed(HERE / f"physnet_embeddings_cells{tag}.npz", emb=cells,
                            dates=np.array([d.isoformat() for d in cube.dates]),
                            lat=lat2[valid], lon=lon2[valid])
    (HERE / f"info{tag}.json").write_text(json.dumps({
        "model": args.model, "days": T, "first": cube.dates[0].isoformat(), "last": cube.dates[-1].isoformat(),
        "dims": 4 * DIM, "blocks": {"emb_0-63": "park mean", "emb_64-127": "park max",
                                    "emb_128-191": "Northern GBR mean", "emb_192-255": "Central GBR mean"},
        "park_cells": int(valid.sum()), "northern_cells": int(zone["N"].sum()), "central_cells": int(zone["C"].sum()),
    }, indent=1))
    print(f"Saved {T} days x {4 * DIM} to {HERE}")


if __name__ == "__main__":
    main()
