# models

Model ada per subfolder, masing-masing dengan `MODEL_CARD.md`, `metrics.json`, dan versi model. `susut-v1/` sudah ada (baseline, data SINTETIS).

Rencana isi:

- `susut-v1/` (ADA) — model prediksi susut per lot (fitur: komoditas, jarak, durasi trip, cuaca).
- `harga-v1/` — model prediksi harga per komoditas per pasar induk.

Setiap model wajib mencantumkan: sumber data, tanggal latih, metrik evaluasi, dan cara serving via API Rust.
