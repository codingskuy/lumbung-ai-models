"""Latih + evaluasi baseline susut pada data SINTETIS, tulis artefak ke models/susut-v1/.

Jalankan: python -m training.susut.train
"""
from __future__ import annotations

import json
import os

from training.common import PENANDA_SINTETIS
from training.susut.data import buat_data_sintetis, split_latih_uji
from training.susut.evaluate import evaluasi
from training.susut.model import BaselineSusut

VERSI_MODEL = "susut-v1.0.0"
KONFIG = {"sortir": "lama_simpan_jam", "jalan": "jarak_km"}


def jalankan(out_dir: str = "models/susut-v1", seed: int = 20261021) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    data = buat_data_sintetis(seed=seed)
    latih, uji = split_latih_uji(data)
    hasil = {"versi_model": VERSI_MODEL, "data": PENANDA_SINTETIS, "seed": seed,
             "n_latih": len(latih), "n_uji": len(uji), "eval": {}}
    for target, fitur in KONFIG.items():
        for nama, f in (("baseline_rata2", None), ("baseline_linier", fitur)):
            m = BaselineSusut(target, f).fit(latih)
            hasil["eval"][f"{target}/{nama}"] = evaluasi(m, uji)
            if nama == "baseline_linier":
                m.simpan(os.path.join(out_dir, f"model-{target}.json"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(hasil, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return hasil


if __name__ == "__main__":
    h = jalankan()
    for k, v in h["eval"].items():
        print(k, "alarm", v["alarm_total"], "tertimbang_aktual%", round(v["susut_tertimbang_aktual_pct"], 3))
