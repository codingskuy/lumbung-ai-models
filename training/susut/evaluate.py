"""Evaluasi baseline susut: MAE, bias, dan alarm terhadap pagu BRD."""
from __future__ import annotations

from training.common import PAGU_JALAN, PAGU_SORTIR, PAGU_TOTAL
from training.susut.data import LotSusut
from training.susut.model import TARGET, BaselineSusut


def pagu(target: str, komoditas: str) -> float:
    return PAGU_SORTIR[komoditas] if target == "sortir" else PAGU_JALAN


def evaluasi(model: BaselineSusut, uji: list[LotSusut]) -> dict:
    y_fn, w_fn = TARGET[model.target]
    per_k: dict[str, dict] = {}
    tp = fp = fn = 0
    tot_w = tot_y = tot_p = 0.0
    for k in sorted({d.komoditas for d in uji}):
        baris = [d for d in uji if d.komoditas == k]
        ys = [y_fn(d) for d in baris]
        ps = [model.prediksi(d) for d in baris]
        ws = [float(w_fn(d)) for d in baris]
        batas = pagu(model.target, k)
        ktp = kfp = kfn = 0
        for y, p in zip(ys, ps):
            aktual, alarm = y > batas, p > batas
            ktp += aktual and alarm
            kfp += (not aktual) and alarm
            kfn += aktual and (not alarm)
        tp, fp, fn = tp + ktp, fp + kfp, fn + kfn
        sw = sum(ws)
        tot_w += sw
        tot_y += sum(w * y for w, y in zip(ws, ys))
        tot_p += sum(w * p for w, p in zip(ws, ps))
        per_k[k] = {
            "n": len(baris),
            "pagu": batas,
            "mae_pp": 100 * sum(abs(y - p) for y, p in zip(ys, ps)) / len(ys),
            "bias_pp": 100 * sum(p - y for y, p in zip(ys, ps)) / len(ys),
            "susut_aktual_tertimbang_pct": 100 * sum(w * y for w, y in zip(ws, ys)) / sw,
            "susut_prediksi_tertimbang_pct": 100 * sum(w * p for w, p in zip(ws, ps)) / sw,
            "lot_melebihi_pagu": ktp + kfn,
            "alarm": {"tp": ktp, "fp": kfp, "fn": kfn},
        }
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {
        "target": model.target,
        "fitur": model.fitur,
        "n_uji": len(uji),
        "per_komoditas": per_k,
        "alarm_total": {"tp": tp, "fp": fp, "fn": fn, "presisi": prec, "recall": rec},
        "susut_tertimbang_aktual_pct": 100 * tot_y / tot_w,
        "susut_tertimbang_prediksi_pct": 100 * tot_p / tot_w,
        "di_bawah_pagu_total": (tot_y / tot_w) < PAGU_TOTAL,
    }
