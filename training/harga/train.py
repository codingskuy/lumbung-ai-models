"""Evaluasi baseline harga pada riwayat SINTETIS, tulis metrik ke models/harga-v1/.

Jalankan: python -m training.harga.train
"""
from __future__ import annotations

import json
import os

from training.common import PENANDA_SINTETIS
from training.harga.data import buat_riwayat_sintetis
from training.harga.evaluate import evaluasi
from training.harga.model import STATUS_KELUARAN, BaselineHarga

VERSI_MODEL = "harga-v1.0.0"
METODE_RILIS = "ema"  # MAPE rata-rata terendah pada data sintetis
KANDIDAT = (BaselineHarga("naif"), BaselineHarga("ma", jendela=4), BaselineHarga("ema", alpha=0.5))


def jalankan(out_dir: str = "models/harga-v1", seed: int = 20261021) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    riwayat = buat_riwayat_sintetis(seed=seed)
    hasil = {"versi_model": VERSI_MODEL, "data": PENANDA_SINTETIS, "seed": seed,
             "status_keluaran": STATUS_KELUARAN, "metode_rilis": METODE_RILIS,
             "n_survei": len(riwayat), "eval": {m.metode: evaluasi(m, riwayat) for m in KANDIDAT}}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, sort_keys=True)
        f.write("\n")
    return hasil


if __name__ == "__main__":
    for m, v in jalankan()["eval"].items():
        print(m, {k: (round(x["mae_rp"]), round(x["mape_pct"], 2)) for k, x in v["per_komoditas"].items()})
