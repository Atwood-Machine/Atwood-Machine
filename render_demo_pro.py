#!/usr/bin/env python3
import os, glob, argparse, time, math, subprocess, shlex
from typing import List
import cv2
import numpy as np

W, H = 1280, 720
FPS_DEFAULT = 24

# ---------- Utilities ----------
def to_qt_safe(inp, outp, fps=24):
    cmd = (
        f'ffmpeg -y -i {shlex.quote(inp)} '
        f'-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p '
        f'-movflags +faststart -r {int(fps)} -an {shlex.quote(outp)}'
    )
    print("[ffmpeg]", cmd)
    subprocess.run(cmd, check=True, shell=True)

def load_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.*")))
    else:
        files = [input_path]
    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not images:
        raise FileNotFoundError("No images found. Provide a folder or image file.")
    print(f"[load] Found {len(images)} image(s).")
    return images

def letterbox(img):
    ih, iw = img.shape[:2]
    r = min(W/iw, H/ih)
    nw, nh = int(iw*r), int(ih*r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = (W-nw)//2, (H-nh)//2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def add_watermark(frame, text="SAMPLE • DO NOT DISTRIBUTE"):
    overlay = frame.copy()
    pad = 16
    scale = 0.7
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x2, y2 = W - pad, H - pad
    x1, y1 = x2 - tw - 2*pad, y2 - th - 2*pad
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
    out = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(out, text, (x1+pad, y2-pad),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick, cv2.LINE_AA)
    return out

def diag_streak_overlay(t, speed=60, width=220, alpha=0.10):
    """Animated diagonal light streaks."""
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    offset = int((t*speed) % (width*2))
    color = (255, 255, 255)
    for k in range(-W, W+H, width*2):
        p = k + offset
        pts = np.array([
            [p, 0], [p+width, 0], [p-width+H, H], [p-2*width+H, H]
        ])
        cv2.fillConvexPoly(overlay, pts, color)
    return overlay, alpha

def ease_out_cubic(x):  # nicer move-in/out
    return 1 - pow(1-x, 3)

# ---------- Effects ----------
def intro_slate(head="Your Free Sample", sub="Auto-rendered preview", secs=2.0, fps=FPS_DEFAULT, brand=(23,124,78)):
    frames = []
    n = int(secs*fps)
    for i in range(n):
        t = i / max(n-1,1)
        base = np.zeros((H, W, 3), dtype=np.uint8)
        # brand bar swipe
        w = int(W * ease_out_cubic(t))
        cv2.rectangle(base, (0,0), (w, H), brand, -1)

        # text fade/move-in
        txt = base.copy()
        alpha = min(1.0, t*1.5)
        y = int(H*0.42 - (1-t)*40)
        cv2.putText(txt, head, (80, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(txt, sub, (82, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230,230,230), 2, cv2.LINE_AA)

        frame = cv2.addWeighted(txt, alpha, base, 1-alpha, 0)
        # subtle vignette
        frame = add_vignette(frame, strength=0.33)
        frames.append(frame)
    return frames

def add_vignette(img, strength=0.25):
    kernel_x = cv2.getGaussianKernel(W, int(W*strength))
    kernel_y = cv2.getGaussianKernel(H, int(H*strength))
    mask = kernel_y * kernel_x.T
    mask = mask / mask.max()
    vignette = np.empty_like(img)
    for c in range(3):
        vignette[:,:,c] = img[:,:,c] * mask
    return vignette

def parallax_segment(img_bgr, seg_secs=3.0, fps=FPS_DEFAULT, zoom=(1.0,1.08), pan_px=(0, 40)):
    """Subtle zoom + vertical pan + moving light streak overlay."""
    base = letterbox(img_bgr)
    frames = []
    n = int(seg_secs*fps)
    for i in range(n):
        t = i / max(n-1,1)
        z = zoom[0] + (zoom[1]-zoom[0]) * t
        # resize
        nh, nw = int(H*z), int(W*z)
        scaled = cv2.resize(base, (nw, nh), interpolation=cv2.INTER_CUBIC)
        # center crop with vertical pan
        x0 = (nw - W)//2
        y_shift = int(pan_px[1] * (t - 0.5) * 2)  # up then down feel
        y0 = max(0, (nh - H)//2 + y_shift)
        y0 = min(nh - H, y0)
        crop = scaled[y0:y0+H, x0:x0+W]

        # overlay animated streaks
        streaks, a = diag_streak_overlay(t, speed=80, width=200, alpha=0.12)
        crop = cv2.addWeighted(streaks, a, crop, 1-a, 0)
        crop = add_vignette(crop, strength=0.22)
        frames.append(crop)
    return frames

def crossfade(a, b, frames=10):
    out = []
    for i in range(frames):
        t = (i+1)/frames
        mix = cv2.addWeighted(b, t, a, 1-t, 0)
        out.append(mix)
    return out

def end_card(cta="Want the full version?", sub="Reply to this email to proceed.", secs=2.0, fps=FPS_DEFAULT, brand=(23,124,78)):
    frames = []
    n = int(secs*fps)
    for i in range(n):
        t = i / max(n-1,1)
        base = np.zeros((H, W, 3), dtype=np.uint8)
        # gradient bg
        for y in range(H):
            v = int(25 + 25*(y/H))
            base[y,:] = (v, v, v)
        # brand border
        cv2.rectangle(base, (12,12), (W-12, H-12), brand, 3)
        # text pop-in
        alpha = min(1.0, t*1.8)
        y = int(H*0.45 - (1-t)*20)
        txt = base.copy()
        cv2.putText(txt, cta, (120, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(txt, sub, (120, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235,235,235), 2, cv2.LINE_AA)
        frame = cv2.addWeighted(txt, alpha, base, 1-alpha, 0)
        frames.append(frame)
    return frames

# ---------- Render ----------
def render_video(images: List[str], out_path: str, total_seconds=12, fps=FPS_DEFAULT,
                 brand_rgb=(23,124,78), wm_text="SAMPLE • DO NOT DISTRIBUTE"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[render] {out_path} | duration {total_seconds}s @ {fps}fps")

    # timing layout: intro(2s) + per-image + end(2s) + transitions
    intro_s = 2.0
    end_s = 2.0
    trans_f = 8  # frames per crossfade
    remain = max(1.0, total_seconds - intro_s - end_s)
    per_img = remain / max(1, len(images))

    frames = []
    # Intro slate
    frames += intro_slate(head="Your Free Sample", sub="Auto-rendered preview", secs=intro_s, fps=fps, brand=brand_rgb)

    # Segments
    prev_last = None
    for idx, p in enumerate(images):
        img = cv2.imread(p)
        if img is None: 
            print(f"[warn] unreadable: {p}")
            continue
        seg = parallax_segment(img, seg_secs=per_img, fps=fps, zoom=(1.0,1.10), pan_px=(0, 60))
        # transition
        if prev_last is not None and seg:
            frames += crossfade(prev_last, seg[0], frames=trans_f)
        frames += seg
        if seg:
            prev_last = seg[-1]

    # End card
    end = end_card(cta="Like what you see?", sub="Reply to this email for the full render.", secs=end_s, fps=fps, brand=brand_rgb)
    if prev_last is not None:
        frames += crossfade(prev_last, end[0], frames=trans_f)
    frames += end

    # Ensure length roughly matches total_seconds (± a little due to integer rounding)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    try:
        for i, f in enumerate(frames):
            f = add_watermark(f, wm_text)
            vw.write(f)
    finally:
        vw.release()
    print(f"[done] Wrote {out_path} ({len(frames)} frames ~= {len(frames)/fps:.2f}s).")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Flashy sample renderer (720p, QuickTime-safe re-encode)")
    ap.add_argument("--input", required=True, help="Folder with images OR single image")
    ap.add_argument("--out", default="outputs/sample_video.mp4", help="Output mp4 path")
    ap.add_argument("--seconds", type=float, default=12.0, help="Total video length")
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT, help="Frames per second")
    ap.add_argument("--brand", default="#177C4E", help="Brand color hex (e.g., #177C4E)")
    ap.add_argument("--watermark", default="SAMPLE • DO NOT DISTRIBUTE", help="Watermark text")
    args = ap.parse_args()

    # parse brand color
    hexv = args.brand.lstrip("#")
    brand_rgb = tuple(int(hexv[i:i+2], 16) for i in (0,2,4))[::-1]  # BGR for OpenCV input order
    brand_rgb = (brand_rgb[2], brand_rgb[1], brand_rgb[0])  # back to RGB for our drawing

    images = load_images(args.input)
    t0 = time.time()
    render_video(images, args.out, total_seconds=args.seconds, fps=args.fps, brand_rgb=(23,124,78), wm_text=args.watermark)

    # QuickTime-safe copy
    qt_out = args.out.replace(".mp4", "_qt.mp4")
    to_qt_safe(args.out, qt_out, fps=args.fps)
    print(f"[elapsed] {time.time()-t0:.2f}s total. QuickTime version: {qt_out}")

if __name__ == "__main__":
    main()
