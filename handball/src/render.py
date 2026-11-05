"""The export video (clip on top, scoreboard and goal chart below) and plots.png."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .ball import smoothed
from .shots import COLS, ROWS
from .skeleton import FACE_KPTS, SKELETON, edge_part
from .text import resolve_font

# The palette shared with the other demos, as RGB.
BG = (0, 0, 0)
FG = (242, 244, 247)
MUTED = (139, 145, 153)
GOAL = (81, 207, 102)
MISS = (255, 107, 107)
BALL = (255, 146, 43)
FRAME = (235, 235, 235)   # posts and bar
GRID = (90, 96, 104)

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"), BarColumn(), TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn(),
)


def _bgr(rgb) -> tuple[int, int, int]:
    return tuple(int(c) for c in rgb[::-1])


def _p(v) -> tuple[int, int]:
    return int(round(v[0])), int(round(v[1]))


@lru_cache(maxsize=32)
def _face(path: str, index: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px, index=index)


def _text(img: np.ndarray, items, font) -> np.ndarray:
    """Draw ``(text, (x, y), px, rgb, anchor)`` items; ``anchor`` is PIL's (e.g. "lm")."""
    if not items:
        return img
    if font is None:
        for text, (x, y), px, rgb, anchor in items:
            fs = px / 34
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            x -= {"l": 0, "m": tw // 2, "r": tw}[anchor[0]]
            y += {"t": th, "m": th // 2, "b": 0, "s": 0}[anchor[1]]
            cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        _bgr(rgb), 2, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, px, rgb, anchor in items:
        draw.text(xy, text, font=_face(font[0], font[1], max(8, int(px))), fill=tuple(rgb),
                  anchor=anchor)
    return np.ascontiguousarray(np.asarray(pil)[..., ::-1])


def _draw_skeleton(frame, pts: np.ndarray, *, colors, thick: int, radius: int) -> None:
    ok = np.isfinite(pts).all(axis=1)
    for a, b, _ in SKELETON:
        if a in FACE_KPTS or b in FACE_KPTS or not (ok[a] and ok[b]):
            continue
        cv2.line(frame, _p(pts[a]), _p(pts[b]), colors[edge_part(a, b)], thick, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i] and i not in FACE_KPTS:
            cv2.circle(frame, _p(pts[i]), radius, (255, 255, 255), -1, cv2.LINE_AA)


def _goal_face(img: np.ndarray, box: tuple[int, int, int, int], shots, k: int | None,
               thick: int, font) -> None:
    """A front-on goal with a 3×3 grid and a dot per decided shot (all shots when *k* is None)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    for i in (1, 2):
        cv2.line(img, (x0 + w * i // 3, y0), (x0 + w * i // 3, y1), _bgr(GRID), 1, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + h * i // 3), (x1, y0 + h * i // 3), _bgr(GRID), 1, cv2.LINE_AA)
    cv2.line(img, (x0, y1), (x0, y0), _bgr(FRAME), thick * 2, cv2.LINE_AA)
    cv2.line(img, (x0, y0), (x1, y0), _bgr(FRAME), thick * 2, cv2.LINE_AA)
    cv2.line(img, (x1, y0), (x1, y1), _bgr(FRAME), thick * 2, cv2.LINE_AA)
    cv2.line(img, (x0 - w // 6, y1), (x1 + w // 6, y1), _bgr(MUTED), max(1, thick // 2), cv2.LINE_AA)
    r = max(4, w // 40)
    for s in shots:
        if s.spot is None or s.goal is None or (k is not None and (s.decided is None or s.decided > k)):
            continue
        u, v = np.clip(s.spot[0], -0.15, 1.15), np.clip(s.spot[1], -0.25, 1.1)
        c = (int(x0 + u * w), int(y0 + v * h))
        cv2.circle(img, c, r, _bgr(GOAL if s.goal else MISS), -1, cv2.LINE_AA)
        cv2.circle(img, c, r + 1, (20, 22, 26), 1, cv2.LINE_AA)


class Panel:
    """The scoreboard under the video: goals of shots on the left, the goal chart on the right."""

    def __init__(self, shots, fps: float, width: int, height: int, *, cfg):
        self.shots, self.fps, self.w, self.h = shots, fps, width, height
        self.font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        self.scale = width / 1280
        gh = int(height * 0.62)
        gw = int(gh * 1.5)            # the goal is 3 m × 2 m
        cx = int(width * 0.70)
        top = int(height * 0.2)
        self.box = (cx - gw // 2, top, cx + gw // 2, top + gh)

    def render(self, k: int) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), np.uint8)
        done = [s for s in self.shots if s.decided is not None and s.decided <= k]
        goals = sum(1 for s in done if s.goal)
        attempts = len([s for s in self.shots if s.release <= k])
        _goal_face(img, self.box, self.shots, k, max(1, round(2 * self.scale)), self.font)
        sc = self.scale
        items = [
            ("GOALS", (int(60 * sc), int(self.h * 0.22)), 28 * sc, MUTED, "lm"),
            (f"{goals} / {attempts}", (int(60 * sc), int(self.h * 0.40)), 96 * sc, FG, "lm"),
        ]
        if done:
            items.append((f"{goals / len(done):.0%}", (int(60 * sc), int(self.h * 0.58)),
                          44 * sc, GOAL if goals / len(done) >= 0.5 else MISS, "lm"))
            last = done[-1]
            line = f"#{last.number} {last.label}" + (f" · {last.zone}" if last.goal and last.zone
                                                     else f" · {last.reason}")
            items.append((line, (int(60 * sc), int(self.h * 0.74)), 30 * sc,
                          GOAL if last.goal else MISS, "lm"))
        x0, _, x1, y1 = self.box
        items.append(("where the ball crossed the line", ((x0 + x1) // 2, y1 + int(34 * sc)),
                      20 * sc, MUTED, "mm"))
        return _text(img, items, self.font)


def render(info, *, kp: np.ndarray, goal, track, shots, dst: Path, cfg, console) -> dict:
    """Write the overlaid clip with the panel beneath it."""
    video_h = min(int(cfg.EXPORT_HEIGHT) // 2, info.height) // 2 * 2
    sx = video_h / info.height
    width = int(round(info.width * sx / 2)) * 2
    out_h = 2 * video_h
    panel = Panel(shots, info.fps, width, out_h - video_h, cfg=cfg)

    scale = width / 720
    thick = max(1, round(cfg.LINE_THICKNESS * scale))
    radius = max(1, round(cfg.POINT_RADIUS * scale))
    fps = info.fps
    flash = int(cfg.VERDICT_SECONDS * fps)
    trail = int(cfg.BALL_TRAIL_SECONDS * fps)
    g0 = (int(goal.left * sx), int(goal.top * sx))
    g1 = (int(goal.right * sx), int(goal.bottom * sx))
    bx, by = smoothed(track, max_gap=int(cfg.BALL_FILL_SECONDS * fps))

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, out_h))
    ball_frames = 0
    with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
        task = progress.add_task("render", total=info.n_frames)
        for k in range(info.n_frames):
            ok, src = cap.read()
            if not ok:
                break
            frame = cv2.resize(src, (width, video_h), interpolation=cv2.INTER_AREA)
            pts = kp[k] * sx
            if np.isfinite(pts).any():
                _draw_skeleton(frame, pts, colors=cfg.SKELETON_COLORS, thick=thick, radius=radius)

            # The goal mouth, lit green or red for a moment once a shot is decided.
            verdict = next((s for s in shots if s.decided is not None and s.goal is not None
                            and s.decided <= k < s.decided + flash), None)
            color = _bgr(GOAL if verdict.goal else MISS) if verdict is not None else _bgr(FRAME)
            cv2.rectangle(frame, g0, g1, color, max(2, thick if verdict is not None else thick // 2),
                          cv2.LINE_AA)
            if verdict is not None and verdict.spot is not None:
                u, v = verdict.spot
                c = (int(g0[0] + u * (g1[0] - g0[0])), int(g0[1] + v * (g1[1] - g0[1])))
                cv2.drawMarker(frame, c, color, cv2.MARKER_CROSS, 6 * thick, thick, cv2.LINE_AA)

            # The ball and its trail, from just before the release until the verdict fades.
            active = next((s for s in shots
                           if s.release - int(0.3 * fps) <= k <= (s.decided or s.end) + flash // 2),
                          None)
            if active is not None and np.isfinite(bx[k]):
                ball_frames += 1
                lo = max(active.release - int(0.3 * fps), k - trail)
                path = [(bx[f] * sx, by[f] * sx) for f in range(lo, k + 1) if np.isfinite(bx[f])]
                for (xa, ya), (xb, yb) in zip(path, path[1:]):
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), _bgr(BALL),
                             max(1, thick - 1), cv2.LINE_AA)
                d = track.d[k] if np.isfinite(track.d[k]) else track.diameter
                r = max(3, int((d if np.isfinite(d) else 12) * sx / 2))
                cv2.circle(frame, (int(bx[k] * sx), int(by[k] * sx)), r, _bgr(BALL),
                           max(2, thick - 1), cv2.LINE_AA)

            writer.write(np.vstack([frame, panel.render(k)]))
            progress.update(task, completed=k + 1)
    cap.release()
    writer.release()
    return {"width": width, "height": out_h, "video_height": video_h,
            "ball_frames_drawn": ball_frames}


def plot_summary(shots, goal, frame_bgr: np.ndarray, dst: Path, *, cfg) -> None:
    """Left: every ball path over the still frame, with the goal. Right: the goal chart and zone tally."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="black",
                               gridspec_kw={"width_ratios": [1.3, 1]})
    for ax in (a, b):
        ax.set_facecolor("black")
        ax.axis("off")
    a.imshow(frame_bgr[..., ::-1], alpha=0.55)
    a.add_patch(plt.Rectangle((goal.left, goal.top), goal.width, goal.height, fill=False,
                              ec="white", lw=1.5))
    for s in shots:
        if not s.path:
            continue
        c = "#51cf66" if s.goal else "#ff6b6b" if s.goal is False else "#8b9199"
        xs, ys = [p[1] for p in s.path], [p[2] for p in s.path]
        a.plot(xs, ys, color=c, lw=1.2)
        a.text(xs[0], ys[0], str(s.number), color=c, fontsize=8)
    a.set_title("ball paths", color="white")

    tally = np.zeros((3, 3), int)
    for s in shots:
        if s.goal and s.zone:
            row, col = s.zone.split()
            tally[ROWS.index(row), COLS.index(col)] += 1
    for i in range(3):
        for j in range(3):
            if tally[i, j]:
                b.text((j + 0.5) / 3 * 3, (i + 0.5) / 3 * 2, str(tally[i, j]), color="#2b2f36",
                       fontsize=40, ha="center", va="center", zorder=0)
    b.plot([0, 0, 3, 3], [2, 0, 0, 2], color="white", lw=4)
    for t in (1, 2):
        b.axvline(t, color="#5a6068", lw=0.8, ymin=0.1, ymax=0.9)
        b.plot([0, 3], [t * 2 / 3] * 2, color="#5a6068", lw=0.8)
    for s in shots:
        if s.spot is None or s.goal is None:
            continue
        u, v = np.clip(s.spot[0], -0.15, 1.15), np.clip(s.spot[1], -0.25, 1.1)
        b.scatter(u * 3, v * 2, s=90, color="#51cf66" if s.goal else "#ff6b6b", ec="black", zorder=3)
        b.text(u * 3 + 0.06, v * 2 - 0.06, str(s.number), color="white", fontsize=8, zorder=4)
    goals = sum(1 for s in shots if s.goal)
    b.set_xlim(-0.6, 3.6)
    b.set_ylim(2.5, -0.6)
    b.set_aspect("equal")
    b.set_title(f"{goals} / {len(shots)} goals  (big numbers: goals per zone)", color="white")
    fig.tight_layout()
    fig.savefig(dst, dpi=150, facecolor="black")
    plt.close(fig)


def write_csv(shots, fps: float, dst: Path) -> None:
    import csv

    rows = [s.summary(fps) for s in shots]
    with dst.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["shot"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (";".join(map(str, v)) if isinstance(v, list) else v)
                             for k, v in r.items()})
