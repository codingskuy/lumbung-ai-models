"""Baseline prediksi susut per komoditas (LMB-AI-002).

Per komoditas: rata-rata tertimbang berat (intercept) + opsional regresi linier 1 fitur
(OLS tertimbang, bentuk tertutup). Hanya pustaka standar; serving tetap lewat API Rust
(parameter diekspor ke JSON sebagai artefak yang dapat dibaca Rust).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from training.common import KOMODITAS
from training.susut.data import LotSusut

# target -> (fungsi susut pecahan, berat awal gram int, berat akhir gram int)
TARGET = {
    "sortir": (lambda d: d.susut_sortir, lambda d: d.berat_kotor_g, lambda d: d.berat_bersih_g),
    "jalan": (lambda d: d.susut_jalan, lambda d: d.berat_berangkat_g, lambda d: d.berat_tiba_g),
}


@dataclass
class BaselineSusut:
    target: str = "sortir"            # "sortir" | "jalan"
    fitur: str | None = None          # nama atribut LotSusut, None = rata-rata murni
    koef: dict[str, tuple[float, float]] = field(default_factory=dict)  # komoditas -> (a, b)

    def fit(self, data: list[LotSusut]) -> "BaselineSusut":
        y_fn, w_fn, _ = TARGET[self.target]
        for k in KOMODITAS:
            baris = [d for d in data if d.komoditas == k]
            if not baris:
                raise ValueError(f"tidak ada data latih untuk komoditas {k}")
            w = [float(w_fn(d)) for d in baris]
            y = [y_fn(d) for d in baris]
            sw = sum(w)
            ybar = sum(wi * yi for wi, yi in zip(w, y)) / sw
            if self.fitur is None:
                self.koef[k] = (ybar, 0.0)
                continue
            x = [float(getattr(d, self.fitur)) for d in baris]
            xbar = sum(wi * xi for wi, xi in zip(w, x)) / sw
            sxx = sum(wi * (xi - xbar) ** 2 for wi, xi in zip(w, x))
            b = 0.0 if sxx == 0 else sum(
                wi * (xi - xbar) * (yi - ybar) for wi, xi, yi in zip(w, x, y)) / sxx
            self.koef[k] = (ybar - b * xbar, b)
        return self

    def prediksi(self, d: LotSusut) -> float:
        if d.komoditas not in self.koef:
            raise ValueError(f"komoditas di luar 3 pilot: {d.komoditas}")
        a, b = self.koef[d.komoditas]
        x = float(getattr(d, self.fitur)) if self.fitur else 0.0
        return max(0.0, a + b * x)

    def ke_dict(self) -> dict:
        return {"target": self.target, "fitur": self.fitur,
                "koef": {k: {"intercept": a, "slope": b} for k, (a, b) in self.koef.items()}}

    def simpan(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ke_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def muat(cls, path: str) -> "BaselineSusut":
        with open(path, encoding="utf-8") as f:
            o = json.load(f)
        m = cls(o["target"], o["fitur"])
        m.koef = {k: (v["intercept"], v["slope"]) for k, v in o["koef"].items()}
        return m
