#!/usr/bin/env python3
"""
EYE  ->  HEART   (ASCII / particle animation)

What this is
------------
A procedural ASCII renderer.  A monospace character grid is filled every frame
by evaluating math fields (an eye, then a heart).  Each cell gets a brightness
`v` and a colour; `v` picks a glyph from a density ramp (" .:-=+*#%@" style).

Extras: your name, an 11:11 moment (at exactly 11.11 s), floating lyric lines
synced to an optional song, and a message box at the end.

Timeline (~30 s, same beats as the reference video)
    0.0 -  2.5  window "decodes" - scrambled letters settle into a white text
                wall, the red iris fades in
    2.5 -  8.0  eye idles: gaze wanders, pupil breathes, two blinks
    8.0 - 12.5  camera pushes into the iris, the white letters dissolve and fly
                off as loose glyph particles
   12.2 - 15.0  iris shape (circle SDF) morphs into a heart (implicit curve)
   11.11        "11:11 - make a wish" blooms out of the dissolving iris
   15.0 - 30.0  heart made of red characters with a white scan-band, heartbeat,
                twinkling sparkles, the name typed under the heart, lyric lines
                floating up; from 22 s a message box opens (type + ENTER)

Run
---
    python eye_to_heart.py --name Reem --audio "song.mp3" --lyrics lyrics.txt
    python eye_to_heart.py --export out.mp4 --audio "song.mp3" --message "hi"
    python eye_to_heart.py --still 22 a.png

lyrics.txt  : one line per row, optionally timestamped LRC-style:
                  [00:14.50] your line here
              (timestamps are song time; --audio-start is subtracted)
--audio     : any mp3/ogg/wav you own.  Live playback needs pygame; export
              needs ffmpeg.
Keys (live): type in the box after 22 s, ENTER sends (saved to
             reem_message.txt), F5 restarts, ESC quits

Requires: numpy, pillow   (optional: pygame for a faster live window)
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageFilter, ImageFont, ImageDraw

# --------------------------------------------------------------------------- #
#  settings
# --------------------------------------------------------------------------- #
COLS, ROWS = 120, 52          # character grid
FONT_SIZE = 13                # px
FPS = 30
DURATION = 30.0               # seconds (export length)
INPUT_T = 22.0                # message box opens
NAME = "Reem"

DEFAULT_LYRICS = [            # original lines (not from any song)
    "make a wish, it is 11:11",
    "tiny seeds of light drift past",
    "for {name}",
    "everything soft finds its way home",
    "hold on to the glow",
    "carried by the wind, never lost",
    "the night is humming your name",
    "11:11 - one quiet wish",
]
SEED = 9

LETTERS = " `.-_:~=zxcvunrjftLCJUYXZOQ0mwqpdbkhao#MW&8%B@"
BRACKETS = " .,-_~:;=+!?)(][}{/\\|1lI<>"

FONT_CANDIDATES = [
    "consola.ttf", "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "DejaVuSansMono.ttf", "cour.ttf",
]


# --------------------------------------------------------------------------- #
#  small math helpers
# --------------------------------------------------------------------------- #
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def ss(a, b, x):
    """smoothstep(a, b, x)  (works for a > b too -> inverted)"""
    t = clamp01((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, k):
    return a + (b - a) * k


def ease_io(t):
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
#  glyph atlas
# --------------------------------------------------------------------------- #
BOLD_CANDIDATES = [
    "consolab.ttf", "C:/Windows/Fonts/consolab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "DejaVuSansMono-Bold.ttf",
]


def load_font(size, bold=False):
    for name in (BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


class Atlas:
    def __init__(self, size=FONT_SIZE):
        self.font = load_font(size)
        l, t, r, b = self.font.getbbox("M")
        self.cw = int(round(self.font.getlength("M")))
        self.ch = int(size * 1.0) + 1
        self.chars = [" "] + [chr(c) for c in range(33, 127)]
        self.index = {c: i for i, c in enumerate(self.chars)}
        glyphs = []
        for c in self.chars:
            im = Image.new("L", (self.cw, self.ch), 0)
            if c != " ":
                ImageDraw.Draw(im).text((0, 0), c, font=self.font, fill=255)
            glyphs.append(np.asarray(im, dtype=np.float32) / 255.0)
        self.alpha = np.stack(glyphs)                      # (n, ch, cw)
        self.let = np.array([self.index[c] for c in LETTERS], dtype=np.int32)
        self.brk = np.array([self.index[c] for c in BRACKETS], dtype=np.int32)


# --------------------------------------------------------------------------- #
#  the scene
# --------------------------------------------------------------------------- #
class Scene:
    def __init__(self):
        self.atlas = Atlas()
        self.cw, self.ch = self.atlas.cw, self.atlas.ch
        self.W, self.H = COLS * self.cw, ROWS * self.ch
        self.rng = np.random.default_rng(SEED)

        rr, cc = np.mgrid[0:ROWS, 0:COLS]
        self.row = rr
        self.aspect = self.W / self.H
        # normalised coordinates: y in [-1, 1] (down +), x in [-A, A]
        self.px = ((cc + 0.5) / COLS * 2 - 1) * self.aspect
        self.py = ((rr + 0.5) / ROWS * 2 - 1)
        self.r1 = self.rng.random((ROWS, COLS)).astype(np.float32)
        self.r2 = self.rng.random((ROWS, COLS)).astype(np.float32)
        self.r3 = self.rng.random((ROWS, COLS)).astype(np.float32)
        self.stripe = np.where(rr % 2 == 0, 1.0, 0.72).astype(np.float32)

        # sparkles (pixel space overlay)
        n = 90
        self.sp_x = self.rng.uniform(8, self.W - 8, n)
        self.sp_y = self.rng.uniform(8, self.H - 8, n)
        self.sp_ph = self.rng.uniform(0, 6.283, n)
        self.sp_sp = self.rng.uniform(1.2, 4.0, n)
        self.sp_cross = self.rng.random(n) < 0.16
        self.sp_len = self.rng.uniform(7, 22, n)
        self.sp_gate = self.rng.random(n)          # when it switches on
        self.sp_vx = self.rng.normal(0, 1.2, n)
        self.sp_vy = self.rng.normal(0, 1.2, n)

        self.name = NAME
        self.typed, self.confirmed, self.confirm_t, self.live = "", False, -99.0, False
        self.lyrics = [(12.8 + 2.3 * i, l.format(name=self.name))
                       for i, l in enumerate(DEFAULT_LYRICS)]
        self.mask_1111 = self.text_mask("11:11", 14, ROWS * 0.45, 1.0)
        self.mask_name = self.text_mask(self.name.upper(), 6, ROWS * 0.80, 1.55)
        cols = np.where(self.mask_name.max(0) > 0.05)[0]
        self.name_c0, self.name_c1 = (cols.min(), cols.max()) if len(cols) else (0, 1)
        self.reset()

    def text_mask(self, text, hrows, crow, spacing=1.0):
        size = int(hrows * self.ch * 1.37)
        font = load_font(size, bold=True)
        adv = [font.getlength(c) * spacing for c in text]
        x = (self.W - sum(adv)) / 2
        im = Image.new("L", (self.W, self.H), 0)
        d = ImageDraw.Draw(im)
        for c, a in zip(text, adv):
            try:
                d.text((x, crow * self.ch), c, font=font, fill=255, anchor="lm")
            except Exception:
                d.text((x, crow * self.ch - size / 2), c, font=font, fill=255)
            x += a
        arr = np.asarray(im, dtype=np.float32) / 255.0
        m = arr.reshape(ROWS, self.ch, COLS, self.cw).mean((1, 3))
        return np.clip(m * 2.2, 0, 1)

    def set_lyrics(self, lines):
        self.lyrics = lines

    # ---- particle state ---------------------------------------------------
    def reset(self):
        self.debris = []                 # dicts: x y vx vy life age ch
        self.last_t = 0.0
        self.prng = np.random.default_rng(SEED + 1)

    # ---- timeline ---------------------------------------------------------
    @staticmethod
    def gaze(t):
        keys = [(0, 0.0, 0.0), (2.6, 0.0, 0.0), (2.85, 0.18, 0.03),
                (4.2, 0.18, 0.03), (4.45, -0.22, 0.05), (5.6, -0.22, 0.05),
                (5.85, 0.06, -0.06), (7.2, 0.06, -0.06), (7.6, 0.0, 0.0)]
        gx = gy = 0.0
        for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                k = ease_io((t - t0) / (t1 - t0))
                gx, gy = mix(x0, x1, k), mix(y0, y1, k)
                break
        else:
            gx, gy = keys[-1][1], keys[-1][2]
        calm = 1.0 - ss(7.5, 10.5, t)
        gx += 0.025 * math.sin(t * 1.7) * calm
        gy += 0.015 * math.sin(t * 2.3 + 1) * calm
        return gx, gy

    @staticmethod
    def blink(t):
        o = 1.0
        for tb in (3.5, 6.5):
            d = (t - tb) / 0.28
            if 0 <= d <= 1:
                o = 1.0 - math.sin(d * math.pi) ** 1.5 * 0.97
        return o

    # ---- main field -> character grid --------------------------------------
    def build(self, t):
        A = self.atlas
        px, py = self.px, self.py
        R = np.hypot(px, py)

        gx, gy = self.gaze(t)
        zoom = 1.0 + 1.15 * ease_io((t - 7.6) / 4.4)
        d_sc = ss(8.6, 12.6, t)             # sclera dissolve
        m = ease_io((t - 12.2) / 2.8)       # iris -> heart morph
        opened = self.blink(t)
        fade_in = ss(0.6, 3.0, t)

        # heartbeat
        beat = 0.0
        if t > 15.0:
            ph = ((t - 15.0) * 1.05) % 1.0
            beat = (math.exp(-((ph - 0.06) / 0.055) ** 2)
                    + 0.7 * math.exp(-((ph - 0.27) / 0.07) ** 2))

        # ================= sclera (screen space / zoomed) ==================
        qx, qy = px / zoom, py / zoom
        wx = clamp01(1.0 - (qx / 1.50) ** 2)
        H = 0.66 * opened * wx ** 0.30
        edge = H - np.abs(qy) + 0.10 * (self.r2 - 0.5)
        almond = ss(0.0, 0.10, edge)
        v_scl = 0.38 + 0.62 * clamp01(edge / 0.35) ** 0.5
        v_scl = v_scl * self.stripe
        v_scl *= 1.0 - 0.35 * ss(0.8, 0.2, np.hypot(qx - gx, qy - gy))  # soft shade around iris
        col_scl = np.stack([np.full_like(v_scl, 238), np.full_like(v_scl, 238),
                            np.full_like(v_scl, 244)], -1)
        thr = d_sc * 1.45 - 0.45 * (R / 1.7)
        vis_scl = (self.r1 > thr).astype(np.float32)
        a_scl = almond * vis_scl

        # ================= iris / heart object (object space) ===============
        Rn = mix(0.50 * zoom, 0.50, m) * (1.0 + 0.055 * beat * m)
        cx = gx * zoom * (1.0 - m)
        cy = gy * zoom * (1.0 - m) - 0.16 * m
        ux, uy = (px - cx) / Rn, (py - cy) / Rn
        r = np.hypot(ux, uy)
        th = np.arctan2(uy, ux)

        # circle implicit (<0 inside) and heart implicit (<0 inside)
        f_c = r * r - 1.0
        hx, hy = ux * 0.92, -(uy * 0.92) + 0.12
        f_h = (hx * hx + hy * hy - 1.0) ** 3 - hx * hx * hy ** 3
        f_h = np.clip(f_h, -3, 3)
        f = mix(f_c, f_h, m)
        soft = mix(0.14, 0.05, m)
        obj = ss(soft, -soft, f)

        # --- iris pattern
        pupil_r = 0.36 * (1.0 + 0.07 * math.sin(t * 1.4))
        fib = (0.55 + 0.22 * np.sin(th * 19 + 2.5 * np.sin(th * 3 + t * 0.4) + r * 6)
               + 0.16 * np.sin(th * 37 - r * 9 + t * 0.7))
        ring = ss(0.34, 0.50, r) * ss(1.03, 0.86, r)
        v_iris = fib * (0.35 + 0.65 * ring)
        v_iris *= ss(pupil_r - 0.05, pupil_r + 0.10, r) * 0.9 + 0.10   # dark pupil
        v_iris += 0.30 * ss(0.84, 1.0, r) * ss(1.10, 0.98, r)           # bright limbal rim
        # highlight (spot + horizontal glint)
        hl = np.exp(-(((ux + 0.30) / 0.13) ** 2 + ((uy + 0.30) / 0.09) ** 2))
        glint = np.exp(-(((ux + 0.02) / 0.50) ** 2 + ((uy - 0.02) / 0.045) ** 2)) * ss(0.7, 0.2, r) * 0.30
        white_iris = clamp01(hl * 1.1 + glint)
        v_iris = clamp01(v_iris * fade_in)
        v_iris = clamp01(v_iris + white_iris * fade_in)
        iris_rgb = np.array([205, 52, 44], dtype=np.float32)

        # --- heart pattern (red body + white scan band)
        light = clamp01(0.55 - 0.18 * uy + 0.10 * ux * -1)
        band = np.exp(-(((uy - 0.10) / 0.30) ** 2)) * (0.65 + 0.35 * np.where(self.row % 2 == 0, 1, 0))
        rim = ss(0.0, -0.9, f_h)
        v_heart = clamp01(0.60 + 0.30 * light + 0.30 * band) * (0.70 + 0.30 * rim)
        v_heart = clamp01(v_heart * (1.0 + 0.25 * beat))
        white_heart = clamp01(band * 1.15) * ss(-0.05, -0.5, f_h)
        heart_rgb = np.array([232, 66, 58], dtype=np.float32)

        v_obj = mix(v_iris, v_heart, m)
        white_obj = mix(white_iris, white_heart, m)
        rgb_obj = (iris_rgb * (1 - white_obj[..., None]) + 255.0 * white_obj[..., None])
        rgb_obj = mix(rgb_obj, (heart_rgb * (1 - white_obj[..., None]) +
                                255.0 * white_obj[..., None]), m)

        # eyelids still occlude the iris until it turns into a heart
        lid = mix(almond, 1.0, ss(0.2, 0.9, m))
        a_obj = obj * lid * mix(fade_in, 1.0, m)

        # =================== composite the two layers =======================
        a_obj_c = a_obj
        v = v_scl * a_scl * (1 - a_obj_c) + v_obj * a_obj_c
        col = (col_scl * (a_scl * (1 - a_obj_c))[..., None] +
               rgb_obj * a_obj_c[..., None])
        w = (a_scl * (1 - a_obj_c) + a_obj_c)
        col = col / np.maximum(w, 1e-4)[..., None]
        wsum = np.maximum(w, 1e-4)
        v = v / wsum * np.minimum(w * 1.4, 1.0)

        # --- boot / reveal: scramble then settle -------------------------------
        y01 = (self.py + 1) * 0.5
        rv = t * 1.15 - (y01 * 0.9 + self.r3 * 0.9)
        revealed = rv > 0
        young = revealed & (rv < 0.45)
        v = np.where(revealed, v, 0.0)

        # --- text overlays made of characters: 11:11 and the name ------------
        s1111 = ss(11.05, 11.25, t) * ss(13.4, 12.4, t)
        Mk = self.mask_1111 * s1111
        fr = clamp01((t - 16.0) / 1.6)
        cut = self.name_c0 + fr * (self.name_c1 - self.name_c0 + 1)
        colidx = np.arange(COLS)[None, :]
        nm = self.mask_name * (colidx <= cut) * ss(15.8, 16.2, t)
        edge_glow = np.exp(-((colidx - cut) / 2.0) ** 2) * (fr < 1) * 0.5
        nm = np.clip(nm * (1 + 0.25 * beat) + (self.mask_name > 0.1) * edge_glow, 0, 1)
        Mk = np.maximum(Mk, nm)
        v = np.maximum(v, Mk)
        col = col * (1 - Mk[..., None]) + np.array([255, 238, 238], np.float32) * Mk[..., None]
        force_let = Mk > 0.25

        # --- pick glyphs ------------------------------------------------------
        bracket_p = np.clip(0.18 + 0.82 * np.maximum(a_obj_c * (1 - white_obj), 0) ** 0.5
                            + 0.1 * m, 0, 1)
        use_brk = (self.r2 < bracket_p) & (white_obj < 0.55) & ~force_let
        kL = np.clip((v * (len(LETTERS) - 1) + 0.5).astype(np.int32), 0, len(LETTERS) - 1)
        kB = np.clip((v * (len(BRACKETS) - 1) * 1.1 + 0.5).astype(np.int32), 0, len(BRACKETS) - 1)
        # time-varying jitter (alive look)
        bucket = np.floor(t * 9 + self.r1 * 9)
        j = (np.sin(bucket * 12.9898 + self.r3 * 78.233) * 43758.5453) % 1.0
        flick = j < 0.07
        off = ((j * 997) % 5).astype(np.int32) - 2
        kL = np.clip(kL + np.where(flick, off, 0), 1, len(LETTERS) - 1)
        kB = np.clip(kB + np.where(flick, off, 0), 1, len(BRACKETS) - 1)
        idx = np.where(use_brk, A.brk[kB], A.let[kL])
        idx = np.where(v < 0.045, 0, idx)
        # scramble young cells
        scr = A.let[(j * 1000).astype(np.int32) % len(LETTERS)]
        idx = np.where(young, scr, idx)
        v = np.where(young, np.maximum(v, 0.55 + 0.4 * j), v)

        brightness = np.clip(0.38 + 0.85 * v, 0, 1.2)
        colour = col * brightness[..., None]

        # small captions written straight into the grid
        def put_text(row, text, rgb, a):
            if a < 0.03:
                return
            c0 = (COLS - len(text)) // 2
            for i, ch_ in enumerate(text):
                if ch_ != " ":
                    idx[row, c0 + i] = A.index[ch_]
                    colour[row, c0 + i] = np.array(rgb, np.float32) * a
        put_text(int(ROWS * 0.45) + 9, "m a k e   a   w i s h", (255, 210, 210), s1111)
        put_text(3, "11:11  -  11/11/2026", (255, 200, 200), ss(16.5, 17.5, t) * 0.9)
        return idx, colour, (px, py, R, m, d_sc)

    # ---- debris particles (flying glyphs) ---------------------------------
    def update_debris(self, t, extra):
        dt = max(0.0, min(0.1, t - self.last_t))
        self.last_t = t
        px, py, R, m, d_sc = extra
        if 8.6 < t < 13.0 and dt > 0:
            n = int(260 * dt * math.sin(math.pi * clamp01((t - 8.6) / 4.4)) + 0.5)
            for _ in range(n):
                r_ = self.prng.integers(0, ROWS)
                c_ = self.prng.integers(0, COLS)
                x = (c_ + 0.5) * self.cw
                y = (r_ + 0.5) * self.ch
                dx, dy = x - self.W / 2, y - self.H / 2
                L = math.hypot(dx, dy) + 1e-3
                sp = self.prng.uniform(25, 110)
                self.debris.append(dict(
                    x=x, y=y, vx=dx / L * sp + self.prng.normal(0, 14),
                    vy=dy / L * sp + self.prng.normal(0, 14),
                    life=self.prng.uniform(1.0, 2.6), age=0.0,
                    ch=int(self.prng.integers(1, len(self.atlas.chars))),
                    tint=self.prng.random()))
        alive = []
        for d in self.debris:
            d["age"] += dt
            if d["age"] < d["life"]:
                d["x"] += d["vx"] * dt
                d["y"] += d["vy"] * dt
                d["vx"] *= (1 - 0.35 * dt)
                d["vy"] *= (1 - 0.35 * dt)
                alive.append(d)
        self.debris = alive

    # ---- final image -------------------------------------------------------
    def render(self, t):
        A = self.atlas
        idx, colour, extra = self.build(t)
        self.update_debris(t, extra)

        glyph = A.alpha[idx]                                         # (R,C,ch,cw)
        img = glyph[..., None] * colour[:, :, None, None, :]
        img = img.transpose(0, 2, 1, 3, 4).reshape(ROWS * self.ch, COLS * self.cw, 3)
        img = np.clip(img, 0, 255).astype(np.float32)

        # debris glyphs
        for d in self.debris:
            x, y = int(d["x"]), int(d["y"])
            if 0 <= x < self.W - self.cw and 0 <= y < self.H - self.ch:
                life = d["age"] / d["life"]
                a = (1 - life) ** 1.3 * 0.9
                tone = np.array([235, 235, 240]) if d["tint"] > 0.25 else np.array([210, 70, 60])
                img[y:y + self.ch, x:x + self.cw] += A.alpha[d["ch"]][..., None] * tone * a

        # sparkles
        gate = ss(10.8, 15.2, t)
        if gate > 0:
            for i in range(len(self.sp_x)):
                if self.sp_gate[i] > gate:
                    continue
                x = (self.sp_x[i] + self.sp_vx[i] * t) % self.W
                y = (self.sp_y[i] + self.sp_vy[i] * t) % self.H
                tw = 0.5 + 0.5 * math.sin(t * self.sp_sp[i] + self.sp_ph[i])
                b = (tw ** 2) * 255
                xi, yi = int(x), int(y)
                if self.sp_cross[i]:
                    L = int(self.sp_len[i] * (0.4 + 0.6 * tw))
                    for k in range(-L, L + 1):
                        f = (1 - abs(k) / (L + 1)) ** 1.6
                        for (X, Y) in ((xi + k, yi), (xi, yi + k)):
                            if 0 <= X < self.W and 0 <= Y < self.H:
                                img[Y, X] += b * f
                    img[max(yi - 1, 0):yi + 2, max(xi - 1, 0):xi + 2] += b * 0.7
                else:
                    s = 1 if tw < 0.7 else 2
                    img[yi:yi + s, xi:xi + s] += b * 0.8

        # floating lyric lines
        for k, (t0, text) in enumerate(self.lyrics):
            age = t - t0
            life = 7.0
            if 0 < age < life:
                p_ = age / life
                a = ss(0, 0.12, p_) * ss(1.0, 0.8, p_) * 0.85
                w_ = len(text) * self.cw
                xb = (0.05 + (0.0, 0.52, 0.22, 0.45, 0.10, 0.55)[k % 6]) * self.W
                xb = min(xb, self.W - w_ - 12)
                x = xb + 14 * math.sin(age * 0.9 + k)
                y = self.H * 0.90 - p_ * self.H * 0.82
                self.draw_string(img, text, x, y, (255, 232, 232), a)

        # message box
        u = ss(INPUT_T - 0.5, INPUT_T + 0.5, t)
        if u > 0:
            x0, x1 = int(self.W * 0.16), int(self.W * 0.84)
            y0 = int(self.H - self.ch * 3.4)
            maxc = (x1 - x0) // self.cw - 4
            shown = self.typed[-maxc:]
            if not self.confirmed:
                for yy in (y0, y0 + self.ch + 10):
                    img[yy, x0:x1] += 110 * u
                img[y0:y0 + self.ch + 11, x0] += 110 * u
                img[y0:y0 + self.ch + 11, x1 - 1] += 110 * u
                self.draw_string(img, "your message  >  type, then press ENTER", x0, y0 - self.ch - 4,
                                 (255, 190, 190), 0.55 * u)
                cur = "_" if int(t * 2) % 2 == 0 else " "
                self.draw_string(img, shown + cur, x0 + 10, y0 + 5, (255, 245, 245), u)
            else:
                w_ = len(shown) * self.cw
                self.draw_string(img, shown, (self.W - w_) / 2, y0 + 5, (255, 245, 245), 1.0)

        # bloom (camera-like glow)
        base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        small = base.resize((self.W // 4, self.H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2.2))
        glow = np.asarray(small.resize((self.W, self.H), Image.BILINEAR), dtype=np.float32)
        out = np.clip(np.asarray(base, dtype=np.float32) + glow * 0.9, 0, 255)

        # fade at the very end / start
        out *= 1.0 + 0.35 * math.exp(-((t - 11.11) / 0.07) ** 2)          # 11:11 flash
        out *= 1.0 + 0.30 * math.exp(-max(t - self.confirm_t, 0) / 0.25) * (t >= self.confirm_t)
        fade = ss(0.0, 0.35, t) * (1.0 if self.live else ss(DURATION, DURATION - 0.8, t))
        return np.clip(out * fade, 0, 255).astype(np.uint8)

    def draw_string(self, img, text, x, y, rgb, a):
        if a <= 0.01:
            return
        tone = np.array(rgb, np.float32) * a
        y = int(y)
        if y < 0 or y + self.ch >= self.H:
            return
        for i, c in enumerate(text):
            xx = int(x) + i * self.cw
            if c == " " or xx < 0 or xx + self.cw >= self.W:
                continue
            gi = self.atlas.index.get(c)
            if gi:
                img[y:y + self.ch, xx:xx + self.cw] += self.atlas.alpha[gi][..., None] * tone


# --------------------------------------------------------------------------- #
#  outputs
# --------------------------------------------------------------------------- #
QUOTES = {ord(k): v for k, v in {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                                  "\u2013": "-", "\u2014": "-", "\u2026": "..."}.items()}


def load_lyrics(path, audio_start, name):
    import re
    out, auto = [], 0
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        raw = [l.strip() for l in f]
    for l in raw:
        if not l or l.startswith("#"):
            continue
        l = l.translate(QUOTES)
        m = re.match(r"\[(\d+):(\d+(?:[.:]\d+)?)\]\s*(.*)", l)
        if m:
            tt = int(m.group(1)) * 60 + float(m.group(2).replace(":", ".")) - audio_start
            text = m.group(3)
        else:
            tt, text = 12.8 + 2.3 * auto, l
            auto += 1
        text = text.replace("{name}", name)
        if text.strip():
            out.append((tt, text))
    return out


def make_scene(args):
    global NAME
    NAME = args.name
    sc = Scene()
    if args.lyrics:
        sc.set_lyrics(load_lyrics(args.lyrics, args.audio_start, args.name))
    return sc


def auto_type(sc, t, message):
    """export mode: type the message by itself"""
    if not message:
        return
    t_start = INPUT_T + 1.0
    n = int(max(0.0, t - t_start) / 0.09)
    sc.typed = message[:n]
    done = t_start + len(message) * 0.09 + 0.4
    if t >= done and not sc.confirmed:
        sc.confirmed, sc.confirm_t = True, done


def export_video(path, args):
    sc = make_scene(args)
    n = int(DURATION * FPS)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{sc.W}x{sc.H}", "-r", str(FPS), "-i", "-"]
    if args.audio:
        cmd += ["-ss", str(args.audio_start), "-i", args.audio,
                "-af", f"afade=t=in:d=0.6,afade=t=out:st={DURATION - 2.5}:d=2.5",
                "-c:a", "aac", "-b:a", "192k", "-t", str(DURATION)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        auto_type(sc, t, args.message)
        p.stdin.write(sc.render(t).tobytes())
        if i % 30 == 0:
            print(f"\r{i / n * 100:5.1f}%", end="", flush=True)
    p.stdin.close()
    p.wait()
    print("\rdone -> " + path)


def save_still(t, path, args):
    sc = make_scene(args)
    step, tt = 1 / FPS, 0.0
    while tt < t - 1e-6:
        tt = min(t, tt + step)
        auto_type(sc, tt, args.message)
        img = sc.render(tt)
    Image.fromarray(img).save(path)
    print("saved", path)


def save_message(text):
    with open("reem_message.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("message saved -> reem_message.txt")


def live(args):
    sc = make_scene(args)
    sc.live = True
    state = {"t0": time.time()}
    mixer = None

    def start_audio():
        nonlocal mixer
        if not args.audio:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(args.audio)
            pygame.mixer.music.play(start=args.audio_start)
            mixer = "pygame"
        except Exception:
            try:
                subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                                  "-ss", str(args.audio_start), args.audio])
                mixer = "ffplay"
            except Exception:
                print("audio: install pygame (pip install pygame) or ffmpeg to hear the song")

    def restart():
        state["t0"] = time.time()
        sc.reset()
        sc.typed, sc.confirmed, sc.confirm_t = "", False, -99.0
        start_audio()

    def key(name, ch):
        """shared key handling; returns False to quit"""
        t = time.time() - state["t0"]
        if name == "escape":
            return False
        if name == "f5":
            restart()
        elif t >= INPUT_T and not sc.confirmed:
            if name == "return":
                if sc.typed.strip():
                    sc.confirmed, sc.confirm_t = True, t
                    save_message(sc.typed)
            elif name == "backspace":
                sc.typed = sc.typed[:-1]
            elif ch and len(ch) == 1 and 32 <= ord(ch) < 127 and len(sc.typed) < 90:
                sc.typed += ch
        return True

    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((sc.W, sc.H))
        pygame.display.set_caption("Eye -> Heart")
        clock = pygame.time.Clock()
        start_audio()
        names = {pygame.K_ESCAPE: "escape", pygame.K_F5: "f5",
                 pygame.K_RETURN: "return", pygame.K_KP_ENTER: "return",
                 pygame.K_BACKSPACE: "backspace"}
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN:
                    if not key(names.get(e.key, ""), e.unicode):
                        return
            frame = sc.render(time.time() - state["t0"])
            surf = pygame.image.frombuffer(frame.tobytes(), (sc.W, sc.H), "RGB")
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)
    except ImportError:
        import tkinter as tk
        root = tk.Tk()
        root.title("Eye -> Heart")
        root.configure(bg="black")
        lbl = tk.Label(root, bd=0)
        lbl.pack()
        start_audio()

        def on_key(e):
            nm = {"Escape": "escape", "F5": "f5", "Return": "return",
                  "BackSpace": "backspace"}.get(e.keysym, "")
            if not key(nm, e.char):
                root.destroy()
        root.bind("<Key>", on_key)

        def tick():
            fr = sc.render(time.time() - state["t0"])
            ppm = b"P6 %d %d 255\n" % (sc.W, sc.H) + fr.tobytes()
            img = tk.PhotoImage(data=ppm, format="PPM")
            lbl.configure(image=img)
            lbl.image = img
            root.after(1, tick)
        tick()
        root.mainloop()


BASE = os.path.dirname(os.path.abspath(__file__))


def find_file(given, exts=None, default_name=None):
    """explicit path (cwd or script folder) -> else auto-detect next to the script"""
    if given:
        for c in (given, os.path.join(BASE, given)):
            if os.path.isfile(c):
                return c
        print(f"!! file not found: {given}")
        return None
    for d in (os.getcwd(), BASE):
        if default_name and os.path.isfile(os.path.join(d, default_name)):
            return os.path.join(d, default_name)
        if exts:
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(exts):
                    return os.path.join(d, f)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ASCII eye -> heart animation")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--audio", help="song file you own (mp3/ogg/wav)")
    ap.add_argument("--audio-start", type=float, default=0.0, help="seconds into the song to start")
    ap.add_argument("--lyrics", help="lyrics.txt (plain lines or [mm:ss.xx] LRC lines)")
    ap.add_argument("--duration", type=float, help="export length in seconds")
    ap.add_argument("--message", default="", help="export/still: auto-type this ending message")
    ap.add_argument("--export", metavar="FILE.mp4", help="render to an mp4 (needs ffmpeg)")
    ap.add_argument("--still", nargs=2, metavar=("SECONDS", "FILE.png"))
    a = ap.parse_args()
    a.lyrics = find_file(a.lyrics, default_name="lyrics.txt")
    a.audio = find_file(a.audio, exts=(".mp3", ".wav", ".ogg", ".m4a", ".flac"))
    if a.duration:
        DURATION = a.duration
    print("lyrics :", a.lyrics or "none found (using built-in lines)")
    print("audio  :", a.audio or "none found (silent)")
    if a.export:
        export_video(a.export, a)
    elif a.still:
        save_still(float(a.still[0]), a.still[1], a)
    else:
        live(a)
