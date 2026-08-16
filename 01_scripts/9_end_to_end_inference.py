#!/usr/bin/env python3
"""
ADIM 9: UCTAN UCA CIKARIM (ASAMA 1 + ASAMA 2 ENTEGRASYONU)  -- v4

ASAMA 1 (ogrenilmis): Egitilmis ikili ResNet18 modeli, her panel-kare
  ciftinin "anomalili" olma olasiligini tahmin eder.
ASAMA 2 (geometrik): Komsuluk analizi kodu, anomalili panelleri
  tekil_panel / hat_arizasi / string_arizasi olarak siniflandirir.

GECMIS:
  v2: elongation_ratio PCA yerine cv2.minAreaRect kullanacak sekilde
      degistirildi (buyuk kumelerde PCA guvenilmezdi).
  v3: PANEL_ANOMALI_THRESH degerinin GERCEKTEN kullanildigini dogrulayan
      debug satirlari eklendi (sorunun kod degil, dosya guncelleme
      sureci oldugu boylece kanitlandi).
  v4: minAreaRect de yetersiz cikti - 11 panelli genis ama kosegen
      yonunde yayilan bir hat_arizasi blogu hala "dar dikdortgen"
      testini yanlislikla geciyordu. Cozum: dogrudan GRID SIRA/SUTUN
      SAYISINA bakan bir olcut. String hatti tipik olarak tek
      sira/sutun boyunca ilerler; kompakt bir blok birden fazla
      sira VE sutuna yayilir.

KULLANIM: VIDEO_TAG'i "video1" veya "video2" olarak ayarlayip calistir.
"""
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image
import numpy as np
import cv2

# ---- AYARLAR ----
VIDEO_TAG = "video2"   # "video1" veya "video2"

MODEL_PATH = Path("8_model_binary/best_resnet18_binary.pt")
OUT_DIR    = Path("9_end_to_end")
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 64
PANEL_ANOMALI_THRESH = 0.7
NEIGH_FACTOR = 1.35
ELONGATION_CUT = 1.8   # grid sira/sutun orani - kalibrasyon gerekebilir

print("="*60)
print(f">>> KULLANILAN ESIK DEGERI (PANEL_ANOMALI_THRESH): {PANEL_ANOMALI_THRESH}")
print(f">>> KULLANILAN ELONGATION_CUT: {ELONGATION_CUT}")
print(f">>> Bu script dosyasi   : {Path(__file__).resolve()}")
print("="*60)

CFG = {
    "video1": {
        "crops": Path("4_panel_crops"),
        "template": Path("2_panel_data_manual/panels_manual.json"),
        "frames": Path("1_Extract_frames"),
        "gt_labels": Path("6_labels/panel_labels.csv"),
    },
    "video2": {
        "crops": Path("4_panel_crops_v2"),
        "template": Path("2_panel_data_manual/panels_manual.json"),
        "frames": Path("1_Extract_frames_v2"),
        "gt_labels": Path("6_labels_v2/panel_labels.csv"),
    },
}[VIDEO_TAG]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Video: {VIDEO_TAG} | Cihaz: {device}")


# ===== 1) MODELI YUKLE =====
print("\nModel yukleniyor...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


# ===== 2) KIRPIM VERI SETI =====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CropInferenceDataset(Dataset):
    def __init__(self, crops_dir, transform):
        self.files = sorted(crops_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = read_image(str(path)).float() / 255.0
        if img.shape[0] == 4:
            img = img[:3]
        img = self.transform(img)
        pid = int(path.stem.split("_p")[1])
        return img, pid


infer_ds = CropInferenceDataset(CFG["crops"], eval_transform)
infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Cikarim yapilacak goruntu sayisi: {len(infer_ds)}")


# ===== 3) ASAMA 1: TUM KIRPIMLAR ICIN ANOMALI OLASILIGI =====
print("\nAsama 1: CNN cikarimi yapiliyor...")
panel_probs = defaultdict(list)

with torch.no_grad():
    for imgs, pids in infer_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        for pid, p in zip(pids.numpy(), probs.cpu().numpy()):
            panel_probs[int(pid)].append(float(p))

panel_score = {pid: float(np.percentile(probs, 90))
               for pid, probs in panel_probs.items()}

print(f"Skorlanan panel sayisi: {len(panel_score)}")

anomalili_ids = {pid for pid, s in panel_score.items() if s > PANEL_ANOMALI_THRESH}
print(f"CNN'e gore anomalili panel sayisi: {len(anomalili_ids)}")


# ===== 4) ASAMA 2: KOMSULUK ANALIZI =====
print("\nAsama 2: Komsuluk analizi uygulaniyor...")

panels = {p["panel_id"]: p for p in json.load(open(CFG["template"]))}

all_centers = {}
for pid, p in panels.items():
    pts = np.array(p["points"], np.float32)
    cx, cy = pts.mean(axis=0)
    all_centers[pid] = (float(cx), float(cy))

def nearest_neighbor_dist(pid):
    cx, cy = all_centers[pid]
    d = [np.hypot(cx - ocx, cy - ocy)
         for opid, (ocx, ocy) in all_centers.items() if opid != pid]
    return min(d) if d else 999999

nn_dist = {pid: nearest_neighbor_dist(pid) for pid in panels}

def are_neighbors(a, b):
    cx1, cy1 = all_centers[a]; cx2, cy2 = all_centers[b]
    d = np.hypot(cx1 - cx2, cy1 - cy2)
    lim = (nn_dist[a] + nn_dist[b]) / 2 * NEIGH_FACTOR
    return d < lim

hot_list = sorted(anomalili_ids)
adj = {i: set() for i in hot_list}
for i in range(len(hot_list)):
    for j in range(i + 1, len(hot_list)):
        a, b = hot_list[i], hot_list[j]
        if are_neighbors(a, b):
            adj[a].add(b)
            adj[b].add(a)

seen, clusters = set(), []
for start in hot_list:
    if start in seen:
        continue
    stack, comp = [start], []
    seen.add(start)
    while stack:
        n = stack.pop()
        comp.append(n)
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    clusters.append(sorted(comp))


def elongation_ratio(pids):
    """
    Kumenin kac SIRA ve kac SUTUN kapladigini olcer.
    String hatti tipik olarak TEK sira/sutun boyunca ilerler; kompakt
    bir hat_arizasi blogu ise birden fazla sira VE sutuna yayilir.
    """
    centers = np.array([all_centers[p] for p in pids])
    xs, ys = centers[:, 0], centers[:, 1]

    avg_diag = np.mean([nn_dist[p] for p in pids]) * 1.4

    def count_groups(vals):
        vals = np.sort(vals)
        groups = 1
        for i in range(1, len(vals)):
            if vals[i] - vals[i-1] > avg_diag:
                groups += 1
        return groups

    n_rows = count_groups(ys)
    n_cols = count_groups(xs)

    return max(n_rows, n_cols) / max(min(n_rows, n_cols), 1)


print(f"\nKume analizi ({len(clusters)} kume):")
for comp in clusters:
    if len(comp) >= 2:
        ratio = elongation_ratio(comp)
        tip = "string_arizasi" if ratio > ELONGATION_CUT else "hat_arizasi"
        print(f"    {len(comp)} panel {comp} -> oran={ratio:.2f} -> {tip}")


predicted_label = {pid: "normal" for pid in panels}
for comp in clusters:
    if len(comp) == 1:
        lab = "tekil_panel"
    else:
        ratio = elongation_ratio(comp)
        lab = "string_arizasi" if ratio > ELONGATION_CUT else "hat_arizasi"
    for pid in comp:
        predicted_label[pid] = lab


# ===== 5) GERCEK ETIKETLERLE KARSILASTIR =====
gt = {}
if CFG["gt_labels"].exists():
    for r in csv.DictReader(open(CFG["gt_labels"])):
        gt[int(r["panel_id"])] = r["label"]

out_rows = []
for pid in sorted(panels):
    out_rows.append({
        "panel_id": pid,
        "cnn_score": round(panel_score.get(pid, 0.0), 4),
        "predicted_label": predicted_label[pid],
        "ground_truth_label": gt.get(pid, "?"),
        "match": predicted_label[pid] == gt.get(pid, None),
    })

out_csv = OUT_DIR / f"final_labels_{VIDEO_TAG}.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["panel_id", "cnn_score", "predicted_label",
                                       "ground_truth_label", "match"])
    w.writeheader()
    w.writerows(out_rows)


# ===== 6) DOGRULAMA HARITASI =====
frame_files = sorted(CFG["frames"].glob("frame_*.png"))
bright = [(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).mean(), f) for f in frame_files]
ref = cv2.imread(str(max(bright, key=lambda x: x[0])[1]))

COLOR = {"normal": (0, 200, 0), "tekil_panel": (255, 255, 255),
        "hat_arizasi": (0, 140, 255), "string_arizasi": (0, 0, 255)}

vis = ref.copy()
for pid, p in panels.items():
    pts = np.array(p["points"], np.int32)
    lab = predicted_label[pid]
    col = COLOR[lab]
    th = 1 if lab == "normal" else 3
    cv2.polylines(vis, [pts], True, col, th)
    tl = pts[np.argmin(pts.sum(axis=1))]
    cv2.putText(vis, str(pid), (int(tl[0]) + 4, int(tl[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

y = 26
cv2.rectangle(vis, (8, 8), (330, 132), (0, 0, 0), -1)
cv2.putText(vis, f"CNN + Geometri Tahmini ({VIDEO_TAG})", (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
y += 22
for txt, col in [("normal", COLOR["normal"]), ("tekil_panel", COLOR["tekil_panel"]),
                 ("hat_arizasi", COLOR["hat_arizasi"]), ("string_arizasi", COLOR["string_arizasi"])]:
    cv2.putText(vis, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    y += 26
cv2.imwrite(str(OUT_DIR / f"label_map_predicted_{VIDEO_TAG}.png"), vis)


# ===== OZET =====
print("\n" + "="*62)
print(f"  SISTEM-SEVIYESI SONUC ({VIDEO_TAG})  |  esik={PANEL_ANOMALI_THRESH}"
      f"  |  elongation_cut={ELONGATION_CUT}")
print("="*62)

pred_cnt = Counter(predicted_label.values())
gt_cnt = Counter(gt.values()) if gt else {}

print(f"\n{'sinif':<16}{'tahmin':>10}{'gercek':>10}")
print("-"*36)
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    print(f"{lab:<16}{pred_cnt.get(lab,0):>10}{gt_cnt.get(lab,0):>10}")

if gt:
    matches = sum(1 for r in out_rows if r["match"])
    print(f"\nPanel-bazinda tam eslesme: {matches}/{len(out_rows)} "
          f"({matches/len(out_rows)*100:.1f}%)")

    def is_anom(lab): return lab != "normal"
    binary_matches = sum(1 for r in out_rows
                         if is_anom(r["predicted_label"]) == is_anom(r["ground_truth_label"]))
    print(f"İkili (anomali var/yok) eslesme: {binary_matches}/{len(out_rows)} "
          f"({binary_matches/len(out_rows)*100:.1f}%)")

    print(f"\nUyusmayan panellerin listesi:")
    for r in out_rows:
        if not r["match"]:
            print(f"    panel {r['panel_id']:>3}: tahmin={r['predicted_label']:<16} "
                  f"gercek={r['ground_truth_label']:<16} (cnn_skor={r['cnn_score']:.3f})")

print(f"\n  CSV    : {out_csv}")
print(f"  Harita : {OUT_DIR}/label_map_predicted_{VIDEO_TAG}.png")
print("="*62)