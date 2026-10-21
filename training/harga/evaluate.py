"""Evaluasi walk-forward: tiap minggu t diprediksi dari riwayat < t."""
from __future__ import annotations

from training.common import KOMODITAS
from training.harga.data import SurveiMingguan
from training.harga.model import BaselineHarga


def evaluasi(model: BaselineHarga, riwayat: list[SurveiMingguan], min_riwayat: int = 4) -> dict:
    per_k = {}
    for k in KOMODITAS:
        seri = sorted((s for s in riwayat if s.komoditas == k), key=lambda s: s.tanggal)
        med = [s.median_ref for s in seri]
        if len(med) <= min_riwayat:
            raise ValueError(f"riwayat {k} terlalu pendek")
        galat = [(model.prediksi(k, med[:t]).harga_pasar_rp_per_kg, med[t])
                 for t in range(min_riwayat, len(med))]
        per_k[k] = {
            "n": len(galat),
            "mae_rp": sum(abs(p - a) for p, a in galat) / len(galat),
            "mape_pct": 100 * sum(abs(p - a) / a for p, a in galat) / len(galat),
        }
    return {"metode": model.metode, "per_komoditas": per_k,
            "mape_rata2_pct": sum(v["mape_pct"] for v in per_k.values()) / len(per_k)}
