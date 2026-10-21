"""Konstanta bersama. Angka bisnis HANYA dari dokumen TownHall (AGENTS.md aturan 7, 9)."""

# Komoditas pilot v1.0.0 (KONTEKS-LINTAS-DIVISI §4; FSD/20-model-data: enum cabai, bawang, kangkung).
KOMODITAS = ("cabai", "bawang", "kangkung")

# Pagu susut sortir (BRD BR-005 / AGENTS.md aturan 9), dalam pecahan 0..1.
PAGU_SORTIR = {"cabai": 0.06, "bawang": 0.04, "kangkung": 0.08}
# Pagu susut jalan maksimal per trip.
PAGU_JALAN = 0.02
# Susut total rata-rata tertimbang wajib di bawah 5%.
PAGU_TOTAL = 0.05

PENANDA_SINTETIS = "SINTETIS - bukan data nyata"
