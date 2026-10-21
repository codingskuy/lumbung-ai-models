# Model card — susut-v1 (versi susut-v1.0.0)

> PERINGATAN: seluruh angka di bawah berasal dari data SINTETIS (karangan generator seed tetap), BUKAN data nyata. Tidak ada klaim akurasi untuk operasi Lumbung sampai model dilatih ulang dengan data timbang ganda dari `lumbung-data-pipeline` (ADR-0001).

## Tujuan
Baseline prediksi susut per lot per komoditas pilot (cabai, bawang, kangkung) untuk deteksi dini lot berisiko terhadap pagu BRD (BR-005; AGENTS.md aturan 9). Dua target: susut sortir (kotor -> bersih) dan susut jalan (per trip). Keluaran bersifat anjuran/alarm dini, bukan keputusan pembayaran.

## Model
- `baseline_rata2`: rata-rata susut tertimbang berat per komoditas.
- `baseline_linier` (artefak rilis): regresi linier 1 fitur per komoditas (OLS tertimbang berat). Sortir memakai `lama_simpan_jam`; jalan memakai `jarak_km`.
- Artefak: `model-sortir.json`, `model-jalan.json` (intercept/slope per komoditas, dapat dibaca Rust). Serving hanya lewat API Rust; tidak ada Python di jalur produksi.
- Dilatih: `python -m training.susut.train` (seed 20261021), hanya pustaka standar Python.

## Data
- Sintetis: 200 lot per komoditas (600), pisah stratified 450 latih / 150 uji. Semua berat bilangan bulat gram (bukan kg desimal; skema menolak float/negatif, `berat_*_total_g` di metrik juga int). Setiap baris bertanda `sintetis=True`.
- Skema mengikuti FSD/20 (timbangan KOTOR/BERSIH) dan KONTEKS §6. Parameter generator karangan, bukan statistik nyata dan bukan pagu BRD.
- Data nyata: akan datang dari `lumbung-data-pipeline` (belum tersedia).

## Ambang alarm (dari BRD)
Sortir: cabai 6%, bawang 4%, kangkung 8%. Susut jalan 2% per trip. Total tertimbang < 5%. Alarm: prediksi > pagu.

## Metrik evaluasi (uji sintetis, 150 lot) — lihat `metrics.json`
Sortir, MAE (poin persen): baseline_rata2 cabai 1,005 / bawang 0,595 / kangkung 1,336; baseline_linier 0,794 / 0,543 / 1,151.
Alarm sortir: rata2 recall 0,00 (30 lot melebihi pagu tidak terdeteksi); linier presisi 0,625, recall 0,33 (tp 10, fp 6, fn 20).
Jalan (linier): MAE 0,13 / 0,13 / 0,16 pp; tidak ada lot melebihi pagu 2% di set uji sintetis, sehingga presisi/recall alarm jalan tidak terdefinisi.
Susut sortir tertimbang aktual set uji sintetis 5,06% (tidak lolos <5% pada data sintetis ini; hanya properti data karangan).

## Batas
- Hanya 3 komoditas pilot; komoditas lain ditolak. Prediksi harga 12 komoditas milik Titeny.
- Baseline sederhana; fitur lain (cuaca, durasi trip) belum dipakai. Recall alarm rendah, jangan dipakai sebagai satu-satunya penjaga mutu.
- Hasil di data nyata belum diketahui; wajib dilatih ulang dan dievaluasi ulang sebelum dipakai.
