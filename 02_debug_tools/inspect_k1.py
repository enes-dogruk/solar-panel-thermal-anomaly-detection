#!/usr/bin/env python3
import csv

CSV_PATH = "6_labels_v2/panel_labels.csv"

rows = list(csv.DictReader(open(CSV_PATH)))

def show(ids, title):
    print(f"\n{title}")
    print(f"{'id':>4} {'hot_p90':>8} {'spread_med':>11} {'big_med':>8}")
    print("-" * 36)
    for r in rows:
        if int(r["panel_id"]) in ids:
            print(f"{r['panel_id']:>4} {r['hot_p90']:>8} "
                  f"{r['spread_med']:>11} {r['big_med']:>8}")

k1_ids = {5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 21, 61}
ref_ids = {46, 68, 65}

show(k1_ids, "K1 KUMESI (13 panel):")
show(ref_ids, "REFERANS - bilinen gercek hotspot panelleri:")