import json
from datetime import date

import pytest

from training.harga import train
from training.harga.data import (batas_retensi, buat_riwayat_sintetis, papan_harga_resmi)
from training.harga.evaluate import evaluasi
from training.harga.model import STATUS_KELUARAN, BaselineHarga


def test_papan_resmi_contoh_dokumen_cabai():
    # keuangan/20-papan-harga-mingguan.md: 36000, 34000, 35000 -> 31500
    assert papan_harga_resmi((36_000, 34_000, 35_000)) == 31_500


def test_papan_resmi_dibulatkan_ke_bawah_500():
    assert papan_harga_resmi((30_000, 30_000, 30_000)) == 27_000
    assert papan_harga_resmi((29_900, 30_000, 31_000)) == 26_500  # 29.900*0,9=26.910 -> 26.500
    assert papan_harga_resmi((6_000, 6_000, 6_000)) == 5_000      # 5.400 -> 5.000


def test_papan_resmi_tidak_bergantung_model():
    import inspect
    assert "BaselineHarga" not in inspect.getsource(papan_harga_resmi)


def test_keluaran_model_berstatus_anjuran():
    p = BaselineHarga("naif").prediksi("cabai", [30_000, 31_000])
    assert p.status == STATUS_KELUARAN == "anjuran-pembanding"
    assert p.harga_pasar_rp_per_kg == 31_000 and isinstance(p.harga_pasar_rp_per_kg, int)


def test_ma_dan_ema():
    assert BaselineHarga("ma", jendela=2).prediksi("bawang", [1000, 2000, 3000]).harga_pasar_rp_per_kg == 2500
    assert BaselineHarga("ema", alpha=0.5).prediksi("bawang", [1000, 2000]).harga_pasar_rp_per_kg == 1500


def test_komoditas_luar_pilot_dan_riwayat_kosong():
    with pytest.raises(ValueError):
        BaselineHarga().prediksi("tomat", [1000])
    with pytest.raises(ValueError):
        BaselineHarga().prediksi("cabai", [])


def test_riwayat_sintetis_ditandai_6_bulan_deterministik():
    r = buat_riwayat_sintetis()
    assert r == buat_riwayat_sintetis()
    assert all(s.sintetis for s in r)
    assert len({s.komoditas for s in r}) == 3
    tgl = [s.tanggal for s in r if s.komoditas == "cabai"]
    assert len(tgl) == 26 and all(t.weekday() == 4 for t in tgl)  # Jumat
    assert min(tgl) >= batas_retensi(date(2026, 10, 9)) or (max(tgl) - min(tgl)).days <= 183


def test_batas_retensi_6_bulan():
    assert batas_retensi(date(2026, 10, 9)) == date(2026, 4, 9)
    assert batas_retensi(date(2026, 8, 31)) == date(2026, 2, 28)
    assert batas_retensi(date(2026, 3, 15)) == date(2025, 9, 15)


def test_evaluasi_mae_mape_dengan_tangan():
    # seri konstan -> naif sempurna
    from training.harga.data import SurveiMingguan
    from datetime import timedelta
    r = [SurveiMingguan(k, date(2026, 1, 2) + timedelta(weeks=i), 10_000, 10_000, 10_000)
         for k in ("cabai", "bawang", "kangkung") for i in range(8)]
    e = evaluasi(BaselineHarga("naif"), r)
    assert e["per_komoditas"]["cabai"]["mae_rp"] == 0 and e["per_komoditas"]["cabai"]["n"] == 4


def test_evaluasi_riwayat_pendek_ditolak():
    r = [s for s in buat_riwayat_sintetis(minggu=4)]
    with pytest.raises(ValueError):
        evaluasi(BaselineHarga("naif"), r)


def test_train_menulis_metrik_bertanda_sintetis(tmp_path):
    h = train.jalankan(str(tmp_path))
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert "SINTETIS" in m["data"] and m["status_keluaran"] == "anjuran-pembanding"
    assert set(h["eval"]) == {"naif", "ma", "ema"}


def test_model_card_harga_lengkap():
    card = open("models/harga-v1/MODEL_CARD.md", encoding="utf-8").read()
    assert train.VERSI_MODEL in card and "SINTETIS" in card
    for bagian in ("Tujuan", "Data", "Metrik", "Batas", "anjuran pembanding", "BR-305", "BR-306", "MAPE"):
        assert bagian in card
