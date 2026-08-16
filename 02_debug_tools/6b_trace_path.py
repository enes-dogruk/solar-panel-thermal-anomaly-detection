#!/usr/bin/env python3
"""
TESHIS: 56 ile 100 hangi zincir uzerinden bagli?
BFS ile en kisa yolu bulur, her kenarin mesafe/esik oranini yazdirir
ve zinciri harita uzerinde vurgular.
"""
import cv2, numpy as np, json, csv
from pathlib import Path
from collections import deque

PROFILE  = Path("5_hotspot/panel_profile.csv")
TEMPLATE = Path("2_panel_data_manual/panels_manual.json")
FRAMES   = Path("1_Extract_frames")

HOT_THRESH, MIN_BLOB = 0.01, 1
RATIO = 0.55   # en dusuk oran (hala birlesiyorlardi)

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

# ===== GRAF KUR (kenar mesafeleri ile birlikte) =====
edges = {}   # (a,b) -> mesafe
adj = {i: set() for i in hot_ids}
for i in range(len(hot_ids)):
    for j in range(i+1, len(hot_ids)):
        a, b = hot_ids[i], hot_ids[j]
        cx1, cy1 = centers[a]; cx2, cy2 = centers[b]
        d = np.hypot(cx1-cx2, cy1-cy2)
        lim = (diag[a] + diag[b]) / 2 * RATIO
        if d < lim:
            adj[a].add(b); adj[b].add(a)
            edges[(a, b)] = (d, lim)
            edges[(b, a)] = (d, lim)

# ===== BFS: 56 -> 100 EN KISA YOL =====
start, goal = 56, 100
parent = {start: None}
q = deque([start])
while q:
    n = q.popleft()
    if n == goal:
        break
    for m in adj[n]:
        if m not in parent:
            parent[m] = n
            q.append(m)

if goal not in parent:
    print("56 ve 100 arasinda hicbir yol bulunamadi (RATIO'yu yukselt).")
else:
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()

    print(f"56 -> 100 ZINCIRI ({len(path)} panel):\n  {path}\n")
    print(f"{'kenar':>12} {'mesafe':>8} {'esik':>8} {'oran(d/esik)':>13}")
    print("-"*50)
    for k in range(len(path)-1):
        a, b = path[k], path[k+1]
        d, lim = edges[(a, b)]
        print(f"  {a:>4}-{b:<4}  {d:>8.1f} {lim:>8.1f} {d/lim:>13.2f}")

    # ===== HARITADA ZINCIRI VURGULA =====
    frame_files = sorted(FRAMES.glob("frame_*.png"))
    bright = [(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).mean(), f) for f in frame_files]
    ref = cv2.imread(str(max(bright, key=lambda x: x[0])[1]))

    vis = ref.copy()
    for pid, p in panels.items():
        pts = np.array(p["points"], np.int32)
        cv2.polylines(vis, [pts], True, (0, 200, 0), 1)

    for k in range(len(path)-1):
        a, b = path[k], path[k+1]
        cv2.line(vis, centers[a], centers[b], (0, 0, 255), 2)
    for pid in path:
        pts = np.array(panels[pid]["points"], np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 3)
        tl = pts[np.argmin(pts.sum(axis=1))]
        cv2.putText(vis, str(pid), (int(tl[0])+4, int(tl[1])+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    out_path = Path("6_labels"); out_path.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path / "chain_56_100.png"), vis)
    print(f"\nHarita: {out_path}/chain_56_100.png")