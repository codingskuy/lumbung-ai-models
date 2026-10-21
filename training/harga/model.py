"""Baseline prediksi harga pasar minggu depan per komoditas (LMB-AI-003).

Targetnya median 3 referensi minggu berikutnya. Keluaran berstatus ANJURAN PEMBANDING
(sanity-check survei); papan resmi tetap `papan_harga_resmi` (BR-305).
"""
from __future__ import annotations

from dataclasses import dataclass

from training.common import KOMODITAS

STATUS_KELUARAN = "anjuran-pembanding"


@dataclass(frozen=True)
class Prediksi:
    komoditas: str
    harga_pasar_rp_per_kg: int
    status: str = STATUS_KELUARAN


@dataclass
class BaselineHarga:
    metode: str = "ma"   # "naif" | "ma" (rata-rata bergerak) | "ema"
    jendela: int = 4
    alpha: float = 0.5

    def _prediksi_seri(self, seri: list[int]) -> float:
        if not seri:
            raise ValueError("riwayat kosong")
        if self.metode == "naif":
            return float(seri[-1])
        if self.metode == "ma":
            w = seri[-self.jendela:]
            return sum(w) / len(w)
        if self.metode == "ema":
            e = float(seri[0])
            for v in seri[1:]:
                e = self.alpha * v + (1 - self.alpha) * e
            return e
        raise ValueError(f"metode tidak dikenal: {self.metode}")

    def prediksi(self, komoditas: str, riwayat_median: list[int]) -> Prediksi:
        if komoditas not in KOMODITAS:
            raise ValueError(f"komoditas di luar 3 pilot: {komoditas}")
        return Prediksi(komoditas, round(self._prediksi_seri(riwayat_median)))
