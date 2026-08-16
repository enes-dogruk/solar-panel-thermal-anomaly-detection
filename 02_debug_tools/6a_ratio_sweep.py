#!/usr/bin/env python3
"""
Panel CAPINA ORANLI komsuluk esigi taramasi.
Sabit piksel (v3) perspektif yuzunden yanlis calisiyordu: uzak/kucuk
panellerde ayni px degeri orantisiz buyuk kaliyor, yakinlarda ise az.
Bu script, esigi HER IKI PANELIN KENDI CAPININ ORTALAMASININ YUZDESI
olarak hesaplar ve farkli oranlari dener.
"""
import cv2, numpy as np, json, csv
from pathlib import Path

PROFILE  = Path("5_hotspot/panel_profile.csv")
TEMPLATE = Path("2_panel_data_manual/panels_manual.json")

HOT_THRESH, MIN_BLOB = 0.01, 1

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}
rows = list(csv.DictReader(open(PROFILE)))
for r in rows:
    r["panel_id"], r["hot_p90"], r["nblob_med"] = \
        int(r["panel_id"]), float(r["hot_p90"]), float(r["nblob_med"])
    r["cx"], r["cy"] = int(r["cx"]), int(r["cy"])

hot_ids = sorted({r["panel_id"] for r in rows
                   if r["hot_p90"] > HOT_THRESH and r["nblob_med"] >= MIN_BLOB})
centers = {r["panel_id"]: (r["cx"], r["cy"]) for r in rows if r["panel_id"] in hot_ids}

def panel_diag(pid):
    pts = np.array(panels[pid]["points"], np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(np.hypot(w, h))

diag = {pid: panel_diag(pid) for pid in hot_ids}

def cluster_for(ratio):
    adj = {i: set() for i in hot_ids}
    for i in range(len(hot_ids)):
        for j in range(i+1, len(hot_ids)):
            a, b = hot_ids[i], hot_ids[j]
            cx1, cy1 = centers[a]; cx2, cy2 = centers[b]
            d = np.hypot(cx1-cx2, cy1-cy2)
            lim = (diag[a] + diag[b]) / 2 * ratio
            if d < lim:
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

print(f"{'oran':>6} {'kume':>6} {'en_buyuk':>9} {'tekil':>7}  detay")
print("-"*75)
for ratio in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
    clusters = cluster_for(ratio)
    multi = [c for c in clusters if len(c) >= 2]
    single = [c for c in clusters if len(c) == 1]
    biggest = max((len(c) for c in clusters), default=0)
    sizes = sorted([len(c) for c in multi], reverse=True)
    tag = ""
    # 56 ve 100'un ayni kumede olup olmadigini isaretle
    for c in multi:
        if 56 in c and 100 in c:
            tag = "  <- 56&100 HALA BIRLESIK"
    print(f"{ratio:>6.2f} {len(multi):>6} {biggest:>9} {len(single):>7}  {sizes}{tag}")