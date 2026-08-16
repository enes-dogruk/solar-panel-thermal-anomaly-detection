#!/usr/bin/env python3
"""
ADIM 11: ONNX DONUSUM + OPTIMIZASYON + KARSILASTIRMALI BENCHMARK

Bu script, egitilmis ikili ResNet18 modelini (8_model_binary/best_resnet18_binary.pt)
alip:
  1) ONNX formatina cevirir
  2) PyTorch ile ONNX ciktilarinin AYNI oldugunu dogrular (kritik kontrol -
     dogrulanmadan "cevirdim" demek yeterli degildir)
  3) Graf optimizasyonu uygular (ONNX Runtime'in kendi optimize edicisi)
  4) INT8 kuantizasyonu uygular (dinamik kuantizasyon - kalibrasyon verisi
     gerektirmez, bu olcekte yeterlidir)
  5) UC modeli de (PyTorch, ONNX-fp32, ONNX-int8) TAM TEST SETINDE yeniden
     degerlendirir (macro F1 dahil) - "boyut kucüldu ama dogruluk da dustu mu"
     sorusuna kesin cevap verir
  6) CPU'da hiz karsilastirmasi yapar (uretim ortami genelde CPU oldugu icin)

NOT: Bu adim projeyi sirkete entegre etmek icin degil, "uretime hazirlik
kapasitesini gostermek" amaciyla yapilmistir (bkz. staj raporu, oneriler).
"""
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

# ---- AYARLAR ----
# NOT: Proje klasoru organize_project.py ile duzenlendi, yollar buna gore
# guncellendi (8_model_binary -> 05_models/model_binary, 7_dataset_split ->
# 04_analysis/dataset_split). Ayrica split dosyalarindaki (test.txt) satirlar
# ESKI kirpim klasoru adlarini (4_panel_crops, 4_panel_crops_v2) referans
# veriyor - bu isimler de tasindigi icin PATH_REMAP ile cevriliyor.
PT_MODEL_PATH = Path("05_models/model_binary/best_resnet18_binary.pt")
SPLIT_DIR     = Path("04_analysis/dataset_split")
BASE_DIR      = Path(".")
OUT_DIR       = Path("11_onnx")
OUT_DIR.mkdir(exist_ok=True)

# test.txt icindeki eski klasor adlari -> organize_project.py sonrasi yeni konum
PATH_REMAP = {
    "4_panel_crops":    "03_data/panel_crops_video1",
    "4_panel_crops_v2": "03_data/panel_crops_video2",
}

def remap_path(rel_path_str):
    """test.txt'deki 'eski_klasor/dosya.png' yolunu yeni klasor konumuna cevirir."""
    parts = rel_path_str.split("/", 1)
    if len(parts) == 2 and parts[0] in PATH_REMAP:
        return f"{PATH_REMAP[parts[0]]}/{parts[1]}"
    return rel_path_str

ONNX_FP32_PATH = OUT_DIR / "model_fp32.onnx"
ONNX_OPT_PATH  = OUT_DIR / "model_fp32_optimized.onnx"
ONNX_INT8_PATH = OUT_DIR / "model_int8.onnx"

CLASSES = ["normal", "anomalili"]
BINARY_MAP = {
    "normal": "normal", "tekil_panel": "anomalili",
    "hat_arizasi": "anomalili", "string_arizasi": "anomalili",
}
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

N_BENCHMARK = 200   # hiz olcumu icin kac goruntu kullanilsin


# ============================================================
# 1) PYTORCH MODELINI YUKLE
# ============================================================
print("="*64)
print("  ADIM 11: ONNX DONUSUM + OPTIMIZASYON + BENCHMARK")
print("="*64)

print("\n[1/6] PyTorch modeli yukleniyor...")
device = torch.device("cpu")  # ONNX karsilastirmasi icin CPU'da tutuyoruz
pt_model = models.resnet18(weights=None)
pt_model.fc = nn.Linear(pt_model.fc.in_features, len(CLASSES))
pt_model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=device))
pt_model.eval()   # KRITIK: BatchNorm/Dropout dogru modda olmali
print(f"  Yuklendi: {PT_MODEL_PATH}")
print(f"  PyTorch model boyutu: {PT_MODEL_PATH.stat().st_size / 1e6:.2f} MB")


# ============================================================
# 2) ONNX'E CEVIR
# ============================================================
print("\n[2/6] ONNX'e cevriliyor...")
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

torch.onnx.export(
    pt_model,
    dummy_input,
    str(ONNX_FP32_PATH),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=13,
    dynamo=False,   # yeni (dynamo) motor 'onnxscript' paketi istiyor; klasik
                    # TorchScript-tabanli motoru kullaniyoruz, ekstra bagimlilik
                    # gerekmez ve bu proje olceginde tamamen yeterlidir
)
print(f"  Kaydedildi: {ONNX_FP32_PATH}")
print(f"  ONNX (fp32) boyutu: {ONNX_FP32_PATH.stat().st_size / 1e6:.2f} MB")

try:
    import onnx
    onnx_model = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ONNX graf yapisi gecerli (onnx.checker.check_model basarili).")
except ImportError:
    print("  UYARI: 'onnx' paketi bulunamadi, graf dogrulama atlandi.")
    print("  Kurulum: pip install onnx")


# ============================================================
# 3) ONNX RUNTIME ILE YUKLE + GRAF OPTIMIZASYONU
# ============================================================
print("\n[3/6] ONNX Runtime session'lari hazirlaniyor (graf optimizasyonu ile)...")
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = str(ONNX_OPT_PATH)

ort_session = ort.InferenceSession(str(ONNX_FP32_PATH), sess_options,
                                   providers=["CPUExecutionProvider"])
print(f"  Optimize edilmis graf kaydedildi: {ONNX_OPT_PATH}")
if ONNX_OPT_PATH.exists():
    print(f"  Optimize graf boyutu: {ONNX_OPT_PATH.stat().st_size / 1e6:.2f} MB")


# ============================================================
# 4) DOGRULAMA: PyTorch ve ONNX AYNI SONUCU VERIYOR MU?
# ============================================================
print("\n[4/6] PyTorch vs ONNX cikti dogrulamasi...")

with torch.no_grad():
    pt_out = pt_model(dummy_input).numpy()

ort_out = ort_session.run(None, {"input": dummy_input.numpy()})[0]

max_diff = np.abs(pt_out - ort_out).max()
print(f"  Rastgele girdide maksimum fark: {max_diff:.2e}")
if max_diff < 1e-4:
    print("  DOGRULAMA BASARILI: PyTorch ve ONNX ciktilari eslesiyor.")
else:
    print("  UYARI: Fark beklenenden buyuk, donusumu kontrol et!")


# ============================================================
# 5) INT8 DINAMIK KUANTIZASYON
# ============================================================
print("\n[5/6] INT8 kuantizasyonu uygulaniyor...")
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input=str(ONNX_FP32_PATH),
    model_output=str(ONNX_INT8_PATH),
    weight_type=QuantType.QInt8,
)
print(f"  Kaydedildi: {ONNX_INT8_PATH}")
print(f"  ONNX (int8) boyutu: {ONNX_INT8_PATH.stat().st_size / 1e6:.2f} MB")

int8_session = ort.InferenceSession(str(ONNX_INT8_PATH),
                                    providers=["CPUExecutionProvider"])
opt_session = ort.InferenceSession(str(ONNX_OPT_PATH),
                                   providers=["CPUExecutionProvider"])


# ============================================================
# YARDIMCI: GORUNTU ON ISLEME (PyTorch transform'larinin numpy karsiligi)
# ============================================================
def load_and_preprocess(path):
    """PIL ile okur, 224x224'e olceklendirir, ImageNet normalize eder,
    CHW + batch boyutunda numpy array döndürür (float32)."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0        # HWC, 0-1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD               # normalize
    arr = arr.transpose(2, 0, 1)                             # CHW
    return arr[np.newaxis, :, :, :].astype(np.float32)       # (1,3,224,224)


# ============================================================
# 6) TAM TEST SETINDE DEGERLENDIRME (3 model icin de)
# ============================================================
print("\n[6/6] Test setinde tam degerlendirme + hiz benchmarki...")

test_samples = []
with open(SPLIT_DIR / "test.txt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        path, orig_label = line.split("\t")
        full_path = BASE_DIR / remap_path(path)
        if full_path.exists():
            test_samples.append((full_path, BINARY_MAP[orig_label]))

print(f"  Test seti: {len(test_samples)} goruntu")
if len(test_samples) == 0:
    print("  HATA: Hic gecerli goruntu bulunamadi. Klasor yapisini kontrol et:")
    print(f"    Aranan SPLIT_DIR : {SPLIT_DIR.resolve()}")
    print(f"    PATH_REMAP       : {PATH_REMAP}")
    raise SystemExit(1)


def evaluate_model(predict_fn, samples):
    """predict_fn(np_array) -> logits (1,2) seklinde numpy array dondurmeli."""
    preds, labels = [], []
    for path, true_label in samples:
        x = load_and_preprocess(path)
        logits = predict_fn(x)
        pred_idx = int(np.argmax(logits, axis=1)[0])
        preds.append(pred_idx)
        labels.append(CLASSES.index(true_label))

    preds = np.array(preds)
    labels = np.array(labels)

    per_class = {}
    f1_scores = []
    for i, c in enumerate(CLASSES):
        tp = int(np.sum((preds == i) & (labels == i)))
        fp = int(np.sum((preds == i) & (labels != i)))
        fn = int(np.sum((preds != i) & (labels == i)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
    }


def pt_predict(x):
    with torch.no_grad():
        return pt_model(torch.from_numpy(x)).numpy()

def onnx_fp32_predict(x):
    return opt_session.run(None, {"input": x})[0]

def onnx_int8_predict(x):
    return int8_session.run(None, {"input": x})[0]


print("\n  Degerlendiriliyor: PyTorch (bu biraz zaman alabilir)...")
pt_result = evaluate_model(pt_predict, test_samples)

print("  Degerlendiriliyor: ONNX fp32 (optimize)...")
onnx_fp32_result = evaluate_model(onnx_fp32_predict, test_samples)

print("  Degerlendiriliyor: ONNX int8 (kuantize)...")
onnx_int8_result = evaluate_model(onnx_int8_predict, test_samples)


# ============================================================
# HIZ BENCHMARKI (CPU)
# ============================================================
print(f"\n  Hiz benchmarki calisiliyor ({N_BENCHMARK} goruntu, CPU)...")

bench_samples = test_samples[:N_BENCHMARK]
bench_inputs = [load_and_preprocess(p) for p, _ in bench_samples]

def benchmark(predict_fn, inputs, warmup=10):
    for x in inputs[:warmup]:            # isinma turu (ilk cagrilar yavas olur)
        predict_fn(x)
    start = time.perf_counter()
    for x in inputs:
        predict_fn(x)
    elapsed = time.perf_counter() - start
    return elapsed / len(inputs) * 1000  # ms/goruntu

pt_ms   = benchmark(pt_predict, bench_inputs)
onnx_ms = benchmark(onnx_fp32_predict, bench_inputs)
int8_ms = benchmark(onnx_int8_predict, bench_inputs)


# ============================================================
# SONUC TABLOSU
# ============================================================
print("\n" + "="*64)
print("  KARSILASTIRMALI SONUC")
print("="*64)

sizes = {
    "PyTorch (.pt)":       PT_MODEL_PATH.stat().st_size / 1e6,
    "ONNX fp32 (optimize)": ONNX_OPT_PATH.stat().st_size / 1e6,
    "ONNX int8 (kuantize)": ONNX_INT8_PATH.stat().st_size / 1e6,
}
speeds = {"PyTorch (.pt)": pt_ms, "ONNX fp32 (optimize)": onnx_ms,
         "ONNX int8 (kuantize)": int8_ms}
results = {"PyTorch (.pt)": pt_result, "ONNX fp32 (optimize)": onnx_fp32_result,
          "ONNX int8 (kuantize)": onnx_int8_result}

print(f"\n{'Model':<24}{'Boyut (MB)':>12}{'Hiz (ms/img)':>14}"
      f"{'Accuracy':>10}{'Macro F1':>10}")
print("-"*72)
for name in sizes:
    r = results[name]
    print(f"{name:<24}{sizes[name]:>12.2f}{speeds[name]:>14.2f}"
          f"{r['accuracy']:>10.3f}{r['macro_f1']:>10.3f}")

print(f"\nSinif bazinda F1 karsilastirmasi:")
print(f"{'Model':<24}{'normal F1':>12}{'anomalili F1':>14}")
print("-"*50)
for name in sizes:
    r = results[name]
    print(f"{name:<24}{r['per_class']['normal']['f1']:>12.3f}"
          f"{r['per_class']['anomalili']['f1']:>14.3f}")

# JSON'a kaydet (rapor icin)
summary = {
    "sizes_mb": sizes,
    "speed_ms_per_image": speeds,
    "results": results,
    "n_benchmark_images": N_BENCHMARK,
    "n_test_images": len(test_samples),
}
with open(OUT_DIR / "onnx_comparison_report.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\n  Rapor kaydedildi: {OUT_DIR}/onnx_comparison_report.json")
print(f"  Modeller: {OUT_DIR}/")
print("="*64)