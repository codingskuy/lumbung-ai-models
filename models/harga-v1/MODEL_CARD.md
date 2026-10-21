# Model card — harga-v1 (versi harga-v1.0.0)

> PERINGATAN: seluruh angka di bawah berasal dari riwayat SINTETIS (karangan generator seed tetap), BUKAN harga nyata. Tidak ada klaim akurasi sampai dievaluasi dengan lembar kerja 3 referensi nyata dari `lumbung-data-pipeline`.

## Tujuan
Baseline prediksi harga pasar minggu depan (median 3 referensi) untuk cabai, bawang, kangkung, sebagai sanity-check survei referensi. Keluaran berstatus anjuran pembanding (`status = "anjuran-pembanding"`). Papan harga resmi tetap formula BR-305 (median 3 referensi - 10%, dibulatkan ke bawah kelipatan Rp500; fungsi `papan_harga_resmi`, tidak memanggil model), BUKAN keluaran model.

Harga ke petani per grade (BRD BR-002 beku): Grade A 100% papan, Grade B 85% papan dibulatkan ke bawah kelipatan Rp500 (contoh dokumen: 31.500 -> 26.500), Grade C ditolak (`harga_papan_grade`; bukan keluaran model). Model hanya memprediksi harga pasar (median 3 referensi), bukan harga per grade. Pembayaran memakai berat bersih gram bilangan bulat (`bayar_petani_rp`, rupiah integer, pembulatan ke bawah = asumsi). Semua berat di repo ini gram integer; harga Rp integer per kg.

## Model
- `naif`: median minggu terakhir. `ma`: rata-rata bergerak 4 minggu. `ema`: rata-rata bergerak eksponensial alpha 0,5 (metode rilis `ema`, MAPE rata-rata terendah di data sintetis).
- Tanpa artefak bobot (parameter hiper di kode). Serving hanya lewat API Rust; tidak ada Python di produksi. Versi: harga-v1.0.0.
- Dijalankan: `python -m training.harga.train` (seed 20261021), hanya pustaka standar.

## Data
- Sintetis: 26 survei Jumat (kira-kira 6 bulan) per komoditas, 3 referensi (pasar induk, pasar kota, eceran daring), rupiah integer per kg. Harga dasar generator karangan (bukan harga pasar nyata).
- Retensi data referensi 6 bulan (BR-306): `batas_retensi(hari_ini)` menghitung tanggal paling awal yang wajib tersimpan; penyimpanan nyata adalah tanggung jawab pipeline/Admin Gudang, bukan repo ini.

## Metrik evaluasi (walk-forward, 22 minggu uji per komoditas, data sintetis) — lihat `metrics.json`
MAE (Rp/kg) / MAPE (%) cabai, bawang, kangkung:
- naif: 959 / 2,60; 505 / 1,74; 305 / 4,98
- ma: 1.098 / 3,00; 631 / 2,17; 217 / 3,57
- ema: 1.005 / 2,73; 529 / 1,82; 234 / 3,83
Perbandingan antar metode hanya valid untuk data sintetis ini.

## Batas
- Hanya 3 komoditas pilot; harga 12 komoditas milik Titeny.
- Tidak menggantikan survei 3 referensi maupun formula papan resmi; tidak dipakai untuk pembayaran.
- Data nyata belum tersedia: wajib dievaluasi ulang (MAE/MAPE) sebelum dipakai.
