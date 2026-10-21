"""Riwayat lembar kerja 3 referensi harga mingguan (SINTETIS) + papan harga resmi BR-305.

Uang integer rupiah per kg (KONTEKS §6). Data nyata: lembar kerja Admin Gudang via
lumbung-data-pipeline (belum tersedia). Generator hanya untuk baseline/uji.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

from training.common import KOMODITAS

BUFFER_LOGISTIK_PCT = 10   # BR-305: median 3 referensi - 10%
KELIPATAN_PAPAN = 500      # dibulatkan ke bawah kelipatan Rp500
RETENSI_BULAN = 6          # BR-306


@dataclass(frozen=True)
class SurveiMingguan:
    komoditas: str
    tanggal: date            # Jumat survei
    ref_pasar_induk: int
    ref_pasar_kota: int
    ref_eceran_daring: int
    sintetis: bool = False

    @property
    def refs(self) -> tuple[int, int, int]:
        return (self.ref_pasar_induk, self.ref_pasar_kota, self.ref_eceran_daring)

    @property
    def median_ref(self) -> int:
        return int(median(self.refs))


def papan_harga_resmi(refs: tuple[int, int, int]) -> int:
    """Formula BR-305 (BUKAN keluaran model): median - 10%, bulat ke bawah kelipatan Rp500. Aritmetika integer."""
    med = int(median(refs))
    setelah = med * (100 - BUFFER_LOGISTIK_PCT) // 100
    return setelah // KELIPATAN_PAPAN * KELIPATAN_PAPAN


def batas_retensi(hari_ini: date) -> date:
    """Tanggal paling awal yang wajib masih tersimpan (6 bulan, BR-306)."""
    bulan = hari_ini.month - RETENSI_BULAN
    tahun = hari_ini.year + (bulan - 1) // 12
    bulan = (bulan - 1) % 12 + 1
    hari = hari_ini.day
    while True:
        try:
            return date(tahun, bulan, hari)
        except ValueError:
            hari -= 1


# Parameter SINTETIS (karangan): harga dasar Rp/kg, drift per minggu, noise relatif.
_PARAM = {"cabai": (35_000, 60, 0.04), "bawang": (30_000, 20, 0.03), "kangkung": (6_500, 5, 0.05)}


def buat_riwayat_sintetis(minggu: int = 26, akhir: date = date(2026, 10, 9),
                          seed: int = 20261021) -> list[SurveiMingguan]:
    """`minggu` survei Jumat berakhir pada `akhir` (default 26 minggu ~ 6 bulan)."""
    rng = random.Random(seed)
    out: list[SurveiMingguan] = []
    for k in KOMODITAS:
        dasar, drift, noise = _PARAM[k]
        tren = float(dasar)
        for i in range(minggu):
            tgl = akhir - timedelta(weeks=minggu - 1 - i)
            tren += drift + rng.gauss(0, dasar * noise / 2)
            refs = [round(tren * (1 + rng.gauss(0, noise)) / 100) * 100 for _ in range(3)]
            out.append(SurveiMingguan(k, tgl, *refs, sintetis=True))
    return out
