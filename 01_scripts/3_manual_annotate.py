#!/usr/bin/env python3
"""
MANUEL PANEL ETIKETLEME (degisken kose sayisi)
Panelin kenarlarina istedigin kadar tikla, sonra 'c' ile kapat.

KONTROLLER:
  Sol tik         : nokta ekle
  c               : paneli kapat ve kaydet (en az 3 nokta)
  z               : son noktayi geri al
  u               : son PANELI geri al
  R + R           : HEPSINI SIFIRLA (iki kez basmak gerekir)
  s               : diske kaydet
  q               : kaydet ve cik
  + / -           : zoom
  Sag tik surukle : kaydir
"""
import cv2, numpy as np, json
from pathlib import Path

FRAMES_DIR = Path("1_extracted_frames")
OUT_JSON   = Path("2_panel_data_manual/panels_manual.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

files = sorted(FRAMES_DIR.glob("frame_*.png"))
if not files:
    print("HATA: 1_extracted_frames/ bos.")
    raise SystemExit(1)

base = cv2.imread(str(files[0]))
H, W = base.shape[:2]

panels, cur = [], []
zoom, ox, oy = 1.0, 0, 0
drag = None
confirm_reset = False

if OUT_JSON.exists():
    panels = json.load(open(OUT_JSON))
    print(f"Onceki kayit yuklendi: {len(panels)} panel")


def to_img(x, y):
    """Ekran koordinati -> goruntu koordinati"""
    return [int(x / zoom + ox), int(y / zoom + oy)]


def close_panel():
    """Aktif poligonu panel olarak kaydet."""
    global cur
    if len(cur) < 3:
        print(f"  En az 3 nokta gerekli (su an {len(cur)})")
        return
    panels.append({"panel_id": len(panels), "points": cur.copy()})
    print(f"  Panel {len(panels)-1} kaydedildi ({len(cur)} kose)")
    cur = []


def on_mouse(event, x, y, flags, param):
    global drag, ox, oy
    if event == cv2.EVENT_LBUTTONDOWN:
        cur.append(to_img(x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        drag = (x, y, ox, oy)
    elif event == cv2.EVENT_MOUSEMOVE and drag:
        x0, y0, o0, o1 = drag
        ox = int(o0 - (x - x0) / zoom)
        oy = int(o1 - (y - y0) / zoom)
    elif event == cv2.EVENT_RBUTTONUP:
        drag = None


cv2.namedWindow("Etiketle", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Etiketle", on_mouse)

print("\nPanelin kenarlarina tikla, bitince 'c' ile kapat.")
print("z=nokta geri | u=panel geri | R+R=SIFIRLA | s=kaydet | q=cik | +/-=zoom\n")

while True:
    vis = base.copy()

    # Kaydedilmis paneller
    for p in panels:
        pts = np.array(p["points"], np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        c = pts.mean(axis=0).astype(int)
        cv2.putText(vis, str(p["panel_id"]), tuple(c),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Aktif (henuz kapanmamis) poligon
    for i, p in enumerate(cur):
        cv2.circle(vis, tuple(p), 4, (0, 165, 255), -1)
        if i > 0:
            cv2.line(vis, tuple(cur[i-1]), tuple(p), (0, 165, 255), 2)
    if len(cur) >= 3:
        cv2.line(vis, tuple(cur[-1]), tuple(cur[0]), (0, 120, 200), 1)

    # Zoom / pan
    ox = max(0, min(ox, int(W - W / zoom)))
    oy = max(0, min(oy, int(H - H / zoom)))
    crop = vis[oy:oy + int(H / zoom), ox:ox + int(W / zoom)]
    view = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    # Bilgi cubugu
    cv2.rectangle(view, (0, 0), (W, 34), (0, 0, 0), -1)
    hint = "'c' ile kapat" if len(cur) >= 3 else "en az 3 nokta"
    cv2.putText(view,
                f"Panel: {len(panels)}  |  aktif nokta: {len(cur)} ({hint})  |  zoom {zoom:.1f}x",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if confirm_reset:
        cv2.putText(view, "SIFIRLAMAK ICIN 'R' TEKRAR BAS", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Etiketle", view)

    k = cv2.waitKey(16) & 0xFF

    if k == ord('R'):                      # Shift+R
        if confirm_reset:
            n = len(panels)
            panels.clear()
            cur.clear()
            confirm_reset = False
            print(f"  SIFIRLANDI ({n} panel silindi). 's' ile diske yazmayi unutma.")
        else:
            confirm_reset = True
            print("  Sifirlamak icin 'R' tusuna TEKRAR bas. Iptal icin baska tusa bas.")
        continue

    if k != 255:
        confirm_reset = False              # baska tusa basilirsa onay iptal

    if k == ord('c'):
        close_panel()
    elif k == ord('z') and cur:
        cur.pop()
    elif k == ord('u') and panels:
        panels.pop()
        for i, p in enumerate(panels):
            p["panel_id"] = i
    elif k in (ord('+'), ord('=')):
        zoom = min(zoom * 1.3, 6.0)
    elif k in (ord('-'), ord('_')):
        zoom = max(zoom / 1.3, 1.0)
    elif k in (ord('s'), ord('q')):
        json.dump(panels, open(OUT_JSON, "w"), indent=2)
        print(f"Kaydedildi: {len(panels)} panel -> {OUT_JSON}")
        if k == ord('q'):
            break

cv2.destroyAllWindows()

# Onizleme goruntusu
prev = base.copy()
for p in panels:
    pts = np.array(p["points"], np.int32)
    cv2.polylines(prev, [pts], True, (0, 255, 0), 2)
    c = pts.mean(axis=0).astype(int)
    cv2.putText(prev, str(p["panel_id"]), tuple(c),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

Path("2_debug").mkdir(exist_ok=True)
cv2.imwrite("2_debug/manual_template.png", prev)
print(f"Onizleme -> 2_debug/manual_template.png  ({len(panels)} panel)")