#!/usr/bin/env python3
"""
ADIM 8b: RESNET18 - IKILI (BINARY) SINIFLANDIRMA
Onceki 4 sinifli denemede (8_train_resnet18.py) confusion matrix su gercegi
ortaya cikardi: tekil_panel ile hat_arizasi arasindaki fark PANELIN KENDI
GORUNTUSUNDE YOK - komsu panelin durumuna bagli. Model bunu goremedigi
icin iki sinifi sistematik olarak birbirine karistirdi (hat_arizasi
recall=0.00, tekil_panel f1=0.19).

YENI MIMARI (iki asamali):
  ASAMA 1 (bu script, CNN): "Bu panelde gorsel anomali var mi?"
           normal / anomalili (tekil_panel+hat_arizasi+string_arizasi birlesik)
  ASAMA 2 (zaten yazilmis kod - 6_label_panels*.py'deki komsuluk analizi):
           CNN'in anomalili dedigi panellere komsuluk grafi uygulanir,
           tip (tekil/hat/string) GEOMETRIDEN belirlenir.

Bu tasarim, "gorsel olarak ogrenilebilir olan" (anomali var mi) ile
"uzamsal/geometrik olan" (komsuluk yapisi) gorevlerini doğru bilesenlere
ayirir. Ayrica string_arizasi'nin dengesizlik sorununu da cozer: uc tip
birlesince (33+42+17=92 panel) CNN icin cok daha dengeli bir gorev olusur.
"""
import csv
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image
import numpy as np

# ---- AYARLAR ----
SPLIT_DIR   = Path("7_dataset_split")
BASE_DIR    = Path(".")
OUT_DIR     = Path("8_model_binary")
OUT_DIR.mkdir(exist_ok=True)

# 4 orijinal sinif ikiliye indirgeniyor
BINARY_MAP = {
    "normal": "normal",
    "tekil_panel": "anomalili",
    "hat_arizasi": "anomalili",
    "string_arizasi": "anomalili",
}
CLASSES = ["normal", "anomalili"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 1e-4
PATIENCE    = 6
IMG_SIZE    = 224
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ===== 1) VERI SETI SINIFI (etiketler binary'e cevrilerek okunur) =====
class PanelDataset(Dataset):
    def __init__(self, list_path, base_dir, transform=None):
        self.samples = []
        with open(list_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                path, orig_label = line.split("\t")
                binary_label = BINARY_MAP[orig_label]
                self.samples.append((base_dir / path, CLASS_TO_IDX[binary_label]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = read_image(str(path)).float() / 255.0
        if img.shape[0] == 4:
            img = img[:3]
        if self.transform:
            img = self.transform(img)
        return img, label


# ===== 2) TRANSFORM'LAR =====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ===== 3) VERI YUKLEYICILER =====
train_ds = PanelDataset(SPLIT_DIR / "train.txt", BASE_DIR, train_transform)
val_ds   = PanelDataset(SPLIT_DIR / "val.txt",   BASE_DIR, eval_transform)
test_ds  = PanelDataset(SPLIT_DIR / "test.txt",  BASE_DIR, eval_transform)

print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))


# ===== 4) SINIF AGIRLIKLARI =====
train_labels = [lbl for _, lbl in train_ds.samples]
counts = Counter(train_labels)
print("\nTrain seti sinif dagilimi (binary):")
for i, c in enumerate(CLASSES):
    print(f"  {c:<12}: {counts.get(i, 0)}")

total = sum(counts.values())
class_weights = torch.tensor(
    [total / (len(CLASSES) * counts.get(i, 1)) for i in range(len(CLASSES))],
    dtype=torch.float32
).to(device)
print(f"\nSinif agirliklari (loss icin): {class_weights.cpu().numpy().round(2)}")


# ===== 5) MODEL =====
print("\nResNet18 (ImageNet on-egitimli) yukleniyor...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3)


# ===== 6) DEGERLENDIRME =====
def evaluate(loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # anomalili olasiligi
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(loader.dataset)

    per_class = {}
    f1_scores = []
    for i, c in enumerate(CLASSES):
        tp = int(np.sum((all_preds == i) & (all_labels == i)))
        fp = int(np.sum((all_preds == i) & (all_labels != i)))
        fn = int(np.sum((all_preds != i) & (all_labels == i)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": int(np.sum(all_labels == i))}
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))
    accuracy = float(np.mean(all_preds == all_labels))

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1,
            "per_class": per_class, "confusion_matrix": cm,
            "preds": all_preds, "labels": all_labels, "probs": all_probs}


# ===== 7) EGITIM DONGUSU =====
print("\n" + "="*60)
print("  EGITIM BASLIYOR (BINARY)")
print("="*60)

best_val_f1 = -1.0
epochs_no_improve = 0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_ds)
    val_result = evaluate(val_loader)
    scheduler.step(val_result["macro_f1"])

    print(f"Epoch {epoch:>3}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} | "
          f"val_loss={val_result['loss']:.4f} | "
          f"val_acc={val_result['accuracy']:.3f} | "
          f"val_macro_f1={val_result['macro_f1']:.3f} | "
          f"val_anomali_recall={val_result['per_class']['anomalili']['recall']:.3f}")

    history.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_result["loss"],
                    "val_acc": val_result["accuracy"],
                    "val_macro_f1": val_result["macro_f1"]})

    if val_result["macro_f1"] > best_val_f1:
        best_val_f1 = val_result["macro_f1"]
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUT_DIR / "best_resnet18_binary.pt")
        print(f"  -> Yeni en iyi model kaydedildi (val_macro_f1={best_val_f1:.3f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n{PATIENCE} epoch boyunca iyilesme olmadi, erken durduruluyor.")
            break

json.dump(history, open(OUT_DIR / "training_history.json", "w"), indent=2)


# ===== 8) TEST =====
print("\n" + "="*60)
print("  TEST SETI DEGERLENDIRMESI (en iyi model)")
print("="*60)

model.load_state_dict(torch.load(OUT_DIR / "best_resnet18_binary.pt"))
test_result = evaluate(test_loader)

print(f"\nTest accuracy : {test_result['accuracy']:.3f}")
print(f"Test macro F1 : {test_result['macro_f1']:.3f}\n")

print(f"{'sinif':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
print("-"*52)
for c in CLASSES:
    pc = test_result["per_class"][c]
    print(f"{c:<12}{pc['precision']:>10.3f}{pc['recall']:>10.3f}"
          f"{pc['f1']:>10.3f}{pc['support']:>10}")

print(f"\nConfusion Matrix (satir=gercek, sutun=tahmin):")
print(f"{'':>12}" + "".join(f"{c:>12}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    row = test_result["confusion_matrix"][i]
    print(f"{c:<12}" + "".join(f"{v:>12}" for v in row))

report = {
    "test_accuracy": test_result["accuracy"],
    "test_macro_f1": test_result["macro_f1"],
    "per_class": test_result["per_class"],
    "confusion_matrix": test_result["confusion_matrix"].tolist(),
    "classes": CLASSES,
}
json.dump(report, open(OUT_DIR / "test_report.json", "w"), indent=2)


# ===== 9) GORSEL =====
try:
    import matplotlib.pyplot as plt

    cm = test_result["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title(f"Confusion Matrix - Binary (macro F1={test_result['macro_f1']:.3f})")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)

    plt.colorbar(im, ax=ax, label="Oran (satir bazinda)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nGorsel kaydedildi: {OUT_DIR}/confusion_matrix.png")
except ImportError:
    print("\nmatplotlib bulunamadi, gorsel olusturulamadi.")

print(f"\n{'='*60}")
print(f"  Model     : {OUT_DIR}/best_resnet18_binary.pt")
print(f"  Test rapor: {OUT_DIR}/test_report.json")
print(f"{'='*60}")
print(f"\n  SONRAKI ADIM: Bu modelin 'anomalili' dedigi panellere")
print(f"  6_label_panels*.py'deki komsuluk analizini uygulayarak")
print(f"  tip (tekil_panel/hat_arizasi/string_arizasi) belirlenecek.")