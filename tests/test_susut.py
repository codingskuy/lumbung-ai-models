import json
import os

import pytest

from training.common import PAGU_JALAN, PAGU_SORTIR
from training.susut.data import LotSusut, buat_data_sintetis, split_latih_uji
from training.susut.evaluate import evaluasi
from training.susut.model import BaselineSusut
from training.susut import train


def lot(k="cabai", kotor=100_000, bersih=95_000, **kw):
    base = dict(lot_id="T", komoditas=k, berat_kotor_g=kotor, berat_bersih_g=bersih,
                berat_berangkat_g=bersih, berat_tiba_g=bersih, jarak_km=10.0, lama_simpan_jam=5.0)
    base.update(kw)
    return LotSusut(**base)


def test_pagu_sesuai_brd():
    assert PAGU_SORTIR == {"cabai": 0.06, "bawang": 0.04, "kangkung": 0.08}
    assert PAGU_JALAN == 0.02


def test_susut_dihitung_dari_gram_integer():
    d = lot(kotor=120_000, bersih=114_000)
    assert d.susut_sortir == pytest.approx(0.05)


def test_data_sintetis_ditandai_dan_deterministik():
    a, b = buat_data_sintetis(20), buat_data_sintetis(20)
    assert a == b and all(d.sintetis for d in a)
    assert all(isinstance(d.berat_kotor_g, int) for d in a)


def test_split_tidak_bocor():
    latih, uji = split_latih_uji(buat_data_sintetis(40))
    assert not {d.lot_id for d in latih} & {d.lot_id for d in uji}


def test_fit_rata2_tertimbang():
    data = [lot(kotor=100_000, bersih=90_000), lot(kotor=300_000, bersih=294_000)]
    data += [lot("bawang"), lot("kangkung")]
    m = BaselineSusut("sortir").fit(data)
    # (0.10*100k + 0.02*300k) / 400k = 0.04
    assert m.prediksi(lot()) == pytest.approx(0.04)


def test_komoditas_luar_pilot_ditolak():
    m = BaselineSusut("sortir").fit(buat_data_sintetis(10))
    with pytest.raises(ValueError):
        m.prediksi(lot("tomat"))


def test_fit_gagal_bila_komoditas_kosong():
    with pytest.raises(ValueError):
        BaselineSusut().fit([lot("cabai")])


def test_alarm_memakai_pagu_per_komoditas():
    # kangkung susut 7% di bawah pagu 8%, cabai 7% di atas pagu 6%
    data = [lot("cabai", bersih=93_000), lot("bawang"), lot("kangkung", bersih=93_000)]
    m = BaselineSusut("sortir").fit(data)
    r = evaluasi(m, data)
    assert r["per_komoditas"]["cabai"]["alarm"]["tp"] == 1
    assert r["per_komoditas"]["kangkung"]["alarm"] == {"tp": 0, "fp": 0, "fn": 0}


def test_regresi_linier_menangkap_slope():
    data = []
    for k in ("cabai", "bawang", "kangkung"):
        for jam in (1, 5, 10, 20):
            s = 0.01 + 0.002 * jam
            data.append(lot(k, bersih=round(100_000 * (1 - s)), lama_simpan_jam=float(jam)))
    m = BaselineSusut("sortir", "lama_simpan_jam").fit(data)
    assert m.koef["cabai"][1] == pytest.approx(0.002, abs=1e-4)


def test_simpan_muat_roundtrip(tmp_path):
    m = BaselineSusut("jalan", "jarak_km").fit(buat_data_sintetis(30))
    p = tmp_path / "m.json"
    m.simpan(str(p))
    m2 = BaselineSusut.muat(str(p))
    d = buat_data_sintetis(1)[0]
    assert m2.prediksi(d) == pytest.approx(m.prediksi(d))


def test_train_menulis_artefak_bertanda_sintetis(tmp_path):
    h = train.jalankan(str(tmp_path))
    assert "SINTETIS" in h["data"]
    for f in ("metrics.json", "model-sortir.json", "model-jalan.json"):
        assert os.path.exists(tmp_path / f)
    assert "SINTETIS" in json.loads((tmp_path / "metrics.json").read_text())["data"]


def test_model_card_dan_versi_ada_di_repo():
    card = open("models/susut-v1/MODEL_CARD.md", encoding="utf-8").read()
    assert train.VERSI_MODEL in card and "SINTETIS" in card
    for bagian in ("Tujuan", "Data", "Metrik", "Batas"):
        assert bagian in card


def test_berat_wajib_gram_integer():
    with pytest.raises(TypeError):
        lot(kotor=100.5)          # kg desimal / float ditolak
    with pytest.raises(TypeError):
        lot(bersih=95_000.0)
    with pytest.raises(ValueError):
        lot(kotor=100_000, bersih=100_001)


def test_metrik_berat_integer_gram():
    data = buat_data_sintetis(30)
    r = evaluasi(BaselineSusut("sortir").fit(data), data)
    for v in r["per_komoditas"].values():
        assert isinstance(v["berat_awal_total_g"], int) and isinstance(v["berat_susut_total_g"], int)
    assert r["per_komoditas"]["cabai"]["berat_susut_total_g"] == sum(
        d.berat_kotor_g - d.berat_bersih_g for d in data if d.komoditas == "cabai")
