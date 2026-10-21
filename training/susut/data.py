"""Skema data timbang ganda + generator data SINTETIS.

Berat integer gram (KONTEKS §6). Data nyata datang dari lumbung-data-pipeline;
generator ini hanya untuk kerangka uji/baseline dan SELALU memberi tanda sintetis=True.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from training.common import KOMODITAS


@dataclass(frozen=True)
class LotSusut:
    lot_id: str
    komoditas: str
    berat_kotor_g: int      # timbangan KOTOR (panen)
    berat_bersih_g: int     # total sortir A+B+C (timbangan BERSIH)
    berat_berangkat_g: int  # berat bersih saat berangkat trip
    berat_tiba_g: int       # berat saat tiba di tujuan
    jarak_km: float
    lama_simpan_jam: float  # jeda panen -> sortir
    sintetis: bool = False

    @property
    def susut_sortir(self) -> float:
        return (self.berat_kotor_g - self.berat_bersih_g) / self.berat_kotor_g

    @property
    def susut_jalan(self) -> float:
        return (self.berat_berangkat_g - self.berat_tiba_g) / self.berat_berangkat_g


# Parameter generator SINTETIS (karangan; BUKAN statistik nyata dan BUKAN pagu BRD).
_PARAM = {
    #            dasar sortir, per jam simpan, noise
    "cabai": (0.040, 0.0008, 0.010),
    "bawang": (0.020, 0.0004, 0.008),
    "kangkung": (0.060, 0.0012, 0.015),
}


def buat_data_sintetis(n_per_komoditas: int = 200, seed: int = 20261021) -> list[LotSusut]:
    rng = random.Random(seed)
    out: list[LotSusut] = []
    for k in KOMODITAS:
        dasar, per_jam, noise = _PARAM[k]
        for i in range(n_per_komoditas):
            kotor = rng.randrange(20_000, 200_001, 500)
            jam = rng.uniform(1, 24)
            jarak = rng.uniform(2, 25)
            s_sortir = max(0.0, dasar + per_jam * jam + rng.gauss(0, noise))
            s_jalan = max(0.0, 0.004 + 0.0004 * jarak + rng.gauss(0, 0.002))
            bersih = round(kotor * (1 - s_sortir))
            tiba = round(bersih * (1 - s_jalan))
            out.append(LotSusut(f"SINT-{k[:3].upper()}-{i:04d}", k, kotor, bersih,
                                bersih, tiba, round(jarak, 1), round(jam, 1), True))
    return out


def split_latih_uji(data: list[LotSusut], frac_uji: float = 0.25, seed: int = 7):
    """Pisah acak deterministik per komoditas (stratified)."""
    rng = random.Random(seed)
    latih: list[LotSusut] = []
    uji: list[LotSusut] = []
    for k in KOMODITAS:
        baris = [d for d in data if d.komoditas == k]
        rng.shuffle(baris)
        n_uji = int(len(baris) * frac_uji)
        uji += baris[:n_uji]
        latih += baris[n_uji:]
    return latih, uji
