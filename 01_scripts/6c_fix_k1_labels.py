#!/usr/bin/env python3
"""
DUZELTME: K1 kumesini elle ikiye ayirir.

GEREKCE: K1 (13 panel) otomatik siniflandirmada tek parca "string_arizasi"
olarak isaretlendi (PCA uzama orani 7.21). Ancak referans karenin (frame_00061)
yakinlastirilmis incelemesinde, kumenin GERCEKTE IKI FARKLI ARIZA
MEKANIZMASININ COGRAFI OLARAK IC ICE GECMIS HALI oldugu goruldu:

  - Bazi paneller (5,8,9,10,14,15,61) BAGIMSIZ, YUVARLAK sicak lekeler
    gosteriyor (hot_p90 > 0.3) -> gercek hat_arizasi deseni.
  - Bazi paneller (6,7,11,13,16,21) hicbir bagimsiz leke gostermiyor,
    aralarindan gecen INCE STRING HATTININ zayif izini tasiyor
    (hot_p90 < 0.2, cogu ~0.02-0.06) -> gercek string_arizasi deseni.

Bu ayrim hem gorsel incelemeyle (zoom crop) hem sayisal veriyle (hot_p90
bimodal dagilimi) dogrulandi. Otomatik PCA/sekil analizi bu iki mekanizmayi
ayni baglantili bilesen icinde oldugu icin ayiramadi; bu yuzden elle
duzeltiliyor.

Bu script SADECE panel_labels.csv'deki 'label' sutununu gunceller,
cluster_id/cluster_size gibi diger alanlara dokunmaz (bilgi amacli kalirlar,
artik K1'in "gercek" bir tek kume olmadigini bilerek yorumlanmalidir).
"""
import csv
from pathlib import Path

CSV_PATH = Path("6_labels_v2/panel_labels.csv")
BACKUP_PATH = Path("6_labels_v2/panel_labels_BEFORE_K1_FIX.csv")

# Yeni etiketler (elle, gorsel+istatistiksel incelemeye dayanarak)
NEW_LABELS = {
    5:  "hat_arizasi",
    8:  "hat_arizasi",
    9:  "hat_arizasi",
    10: "hat_arizasi",
    14: "hat_arizasi",
    15: "hat_arizasi",
    61: "hat_arizasi",
    6:  "string_arizasi",
    7:  "string_arizasi",
    11: "string_arizasi",
    13: "string_arizasi",
    16: "string_arizasi",
    21: "string_arizasi",
}

# ===== 1) YEDEK AL =====
rows = list(csv.DictReader(open(CSV_PATH)))
fieldnames = list(rows[0].keys())

with open(BACKUP_PATH, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"Yedek alindi: {BACKUP_PATH}")

# ===== 2) ETIKETLERI GUNCELLE =====
changed = []
for r in rows:
    pid = int(r["panel_id"])
    if pid in NEW_LABELS:
        old = r["label"]
        new = NEW_LABELS[pid]
        if old != new:
            changed.append((pid, old, new))
        r["label"] = new

with open(CSV_PATH, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\nGuncellendi: {CSV_PATH}\n")
print(f"{'panel':>6} {'eski':>16} {'yeni':>16}")
print("-" * 42)
for pid, old, new in sorted(changed):
    print(f"{pid:>6} {old:>16} {new:>16}")

# ===== 3) YENI DAGILIM =====
from collections import Counter
cnt = Counter(r["label"] for r in rows)

print("\n" + "="*50)
print("  GUNCEL SINIF DAGILIMI (K1 duzeltmesinden sonra)")
print("="*50)
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    n = cnt.get(lab, 0)
    print(f"    {lab:<16}: {n:>3} panel  ({n/len(rows)*100:>5.1f}%)")
print(f"    {'TOPLAM':<16}: {len(rows):>3}")
print("="*50)