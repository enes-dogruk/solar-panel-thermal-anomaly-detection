#!/usr/bin/env python3
"""
DILATE_PX taramasi: hangi degerde kume yapisi en 'saglikli' cikiyor?
33 lekeli panel icin farkli genisletme degerlerini dener, kume sayisini
ve en buyuk kumenin boyutunu raporlar.
"""
import cv2, numpy as np, json, csv
from pathlib import Path

PROFILE  = Path("5_hotspot/panel_profile.csv")
TEMPLATE = Path("2_panel_data_manual/panels_manual.json")
FRAMES   = Path("1_Extract_frames")

HOT_THRESH, MIN_BLOB = 0.01, 1

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}
rows = list(csv.DictReader(open(PROFILE)))
for r in rows:
    r["panel_id"], r["hot_p90"], r["nblob_med"] = \
        int(r["panel_id"]), float(r["hot_p90"]), float(r["nblob_med"])

hot_ids = sorted({r["panel_id"] for r in rows
                   if r["hot_p90"] > HOT_THRESH and r["nblob_med"] >= MIN_BLOB})

sample = cv2.imread(str(sorted(FRAMES.glob("frame_*.png"))[0]))
Hc, Wc = sample.shape[:2]

def cluster_for(dilate_px):
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
    masks = {}
    for pid in hot_ids:
        pts = np.array(panels[pid]["points"], np.int32)
        m = np.zeros((Hc, Wc), np.uint8)
        cv2.fillPoly(m, [pts], 255)
        masks[pid] = cv2.dilate(m, K, iterations=1)

    adj = {i: set() for i in hot_ids}
    for i in range(len(hot_ids)):
        for j in range(i+1, len(hot_ids)):
            a, b = hot_ids[i], hot_ids[j]
            if cv2.countNonZero(cv2.bitwise_and(masks[a], masks[b])) > 0:
                adj[a].add(b); adj[b].add(a)

    seen, clusters = set(), []
    for start in hot_ids:
        if start in seen: continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            n = stack.pop(); comp.append(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m); stack.append(m)
        clusters.append(sorted(comp))
    return clusters

print(f"{'px':>4} {'kume':>6} {'en_buyuk':>9} {'tekil':>7}  detay")
print("-"*70)
for px in [1, 2, 3, 4, 5, 6, 8, 10]:
    clusters = cluster_for(px)
    multi = [c for c in clusters if len(c) >= 2]
    single = [c for c in clusters if len(c) == 1]
    biggest = max((len(c) for c in clusters), default=0)
    sizes = sorted([len(c) for c in multi], reverse=True)
    print(f"{px:>4} {len(multi):>6} {biggest:>9} {len(single):>7}  {sizes}")