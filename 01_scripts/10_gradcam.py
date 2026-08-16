#!/usr/bin/env python3
"""
ADIM 10: GRAD-CAM GORSELLESTIRME
Egitilmis ikili ResNet18 modelinin ("normal" / "anomalili") her panel
goruntusunde NEREYE baktigini gosteren isi haritalari uretir.

MANTIK:
  1) Panel goruntusu modelden gecirilir, tahmin edilen sinifin skoru alinir.
  2) Bu skorun, son konvolusyon katmaninin (ResNet18'de layer4) ciktisina
     gore gradyani hesaplanir - "bu katmandaki her ozellik haritasi,
     karari ne kadar etkiledi?" sorusunun cevabi.
  3) Her ozellik haritasi, kendi gradyaninin ortalamasiyla agirliklandirilip
     toplanir -> kaba bir "onem haritasi" (CAM) elde edilir.
  4) Bu harita orijinal goruntu boyutuna buyutulup renkli isi haritasi
     olarak panelin uzerine bindirilir.

CIKTI: Secilen ornek paneller icin [orijinal | isi haritasi | bindirme]
       seklinde kolaj goruntuler, 10_gradcam/ klasorune kaydedilir.
"""
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.io import read_image
import numpy as np
import cv2

# ---- AYARLAR ----
MODEL_PATH  = Path("8_model_binary/best_resnet18_binary.pt")
SPLIT_DIR   = Path("7_dataset_split")
BASE_DIR    = Path(".")
OUT_DIR     = Path("10_gradcam")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["normal", "anomalili"]
BINARY_MAP = {
    "normal": "normal",
    "tekil_panel": "anomalili",
    "hat_arizasi": "anomalili",
    "string_arizasi": "anomalili",
}

IMG_SIZE = 224
N_PER_GROUP = 4   # her kategoriden kac ornek gosterilsin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")


# ===== 1) MODELI YUKLE =====
print("Model yukleniyor...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


# ===== 2) GRAD-CAM ICIN KANCA (HOOK) MEKANIZMASI =====
# ResNet18'in son konvolusyon blogu 'layer4'. Bu katmanin CIKTISINI ve
# ona gelen GRADYANI yakalamak icin ileri/geri kancalar takiyoruz.
activations = {}
gradients = {}

def forward_hook(module, inp, out):
    activations["value"] = out.detach()

def backward_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0].detach()

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


def compute_gradcam(img_tensor, class_idx):
    """
    Tek bir goruntu (1,3,H,W) icin Grad-CAM isi haritasini hesaplar.
    Donen deger: 0-1 arasi normalize edilmis, (H,W) boyutunda numpy dizisi.
    """
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor.requires_grad_(False)

    model.zero_grad()
    output = model(img_tensor)
    score = output[0, class_idx]
    score.backward()

    acts = activations["value"][0]        # (C, h, w)
    grads = gradients["value"][0]          # (C, h, w)

    weights = grads.mean(dim=(1, 2))       # (C,) - her kanalin onem agirligi
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=device)
    for c in range(acts.shape[0]):
        cam += weights[c] * acts[c]

    cam = F.relu(cam)                      # sadece pozitif katkilari al
    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, float(torch.softmax(output, dim=1)[0, class_idx].item())


# ===== 3) GORUNTU HAZIRLAMA =====
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def load_image(path):
    """Ham goruntuyu hem model icin normalize edilmis hem de goruntuleme
    icin 0-255 RGB olarak dondurur."""
    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 4:
        img = img[:3]
    img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False)[0]

    display = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB, 0-255

    norm = img.clone()
    for c in range(3):
        norm[c] = (norm[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    return norm, display


def make_overlay(display_rgb, cam):
    """Isi haritasini orijinal goruntunun uzerine bindirir."""
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.55 * display_rgb + 0.45 * heatmap).astype(np.uint8)
    return heatmap, overlay


# ===== 4) TEST SETINDEN ORNEK PANELLER SEC =====
test_samples = []
with open(SPLIT_DIR / "test.txt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        path, orig_label = line.split("\t")
        binary_label = BINARY_MAP[orig_label]
        test_samples.append((BASE_DIR / path, binary_label, orig_label))

# Once modelin TUM secilen adaylar icin tahminini al (dogru/yanlis ayrimi icin)
print(f"\n{len(test_samples)} test ornegi taraniyor...")

candidates = {"anomalili_dogru": [], "normal_dogru": [], "yanlis_tahmin": []}

for path, true_label, orig_label in test_samples:
    if not path.exists():
        continue
    norm_img, _ = load_image(path)
    with torch.no_grad():
        out = model(norm_img.unsqueeze(0).to(device))
        pred_idx = out.argmax(dim=1).item()
    pred_label = CLASSES[pred_idx]

    if pred_label == true_label == "anomalili" and len(candidates["anomalili_dogru"]) < N_PER_GROUP:
        candidates["anomalili_dogru"].append((path, true_label, orig_label, pred_idx))
    elif pred_label == true_label == "normal" and len(candidates["normal_dogru"]) < N_PER_GROUP:
        candidates["normal_dogru"].append((path, true_label, orig_label, pred_idx))
    elif pred_label != true_label and len(candidates["yanlis_tahmin"]) < N_PER_GROUP:
        candidates["yanlis_tahmin"].append((path, true_label, orig_label, pred_idx))

    if all(len(v) >= N_PER_GROUP for v in candidates.values()):
        break

print("Secilen ornekler:")
for grp, items in candidates.items():
    print(f"  {grp}: {len(items)} ornek")


# ===== 5) HER GRUP ICIN GRAD-CAM KOLAJI URET =====
for grp_name, items in candidates.items():
    if not items:
        continue

    rows = []
    for path, true_label, orig_label, pred_idx in items:
        norm_img, display_rgb = load_image(path)
        cam, score = compute_gradcam(norm_img, pred_idx)
        heatmap, overlay = make_overlay(display_rgb, cam)

        # Uc panel yan yana: orijinal | isi haritasi | bindirme
        trio = np.hstack([display_rgb, heatmap, overlay])
        trio_bgr = cv2.cvtColor(trio, cv2.COLOR_RGB2BGR)

        label_text = f"{path.stem} | gercek={orig_label} | tahmin={CLASSES[pred_idx]} ({score:.2f})"
        banner = np.zeros((28, trio_bgr.shape[1], 3), np.uint8)
        cv2.putText(banner, label_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        rows.append(np.vstack([banner, trio_bgr]))

    collage = np.vstack(rows)
    out_path = OUT_DIR / f"gradcam_{grp_name}.png"
    cv2.imwrite(str(out_path), collage)
    print(f"Kaydedildi: {out_path}")

print(f"\nTum sonuclar: {OUT_DIR}/")
print("Her satirda: [orijinal panel] [Grad-CAM isi haritasi] [bindirilmis hali]")