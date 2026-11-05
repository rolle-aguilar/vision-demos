"""Offline checks for the goal-or-miss logic and the render, on made-up tracks (no gateway calls).

    python -m pytest tests            # from handball/
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ball as ball_mod, render, shots as shots_mod  # noqa: E402
from src.goal import Goal  # noqa: E402

FPS = 30.0
GOAL = Goal(left=600, right=1320, top=300, bottom=780, score=0.9, frame=0)   # 720 × 480 px
AT_GOAL = shots_mod.ball_at_goal_px(GOAL, ball_m=0.185, goal_width_m=3.0)    # ~44 px


def flight(u: float, v: float, n: int = 40, *, start_size: float = 160, back: int = 0):
    """A ball flying from the shooter's hand to (u, v) on the goal face, shrinking as it goes,
    then *back* frames bouncing back towards the camera."""
    x = np.full(n + back + 10, np.nan)
    y, d = x.copy(), x.copy()
    x0, y0 = 960, 900
    x1, y1 = GOAL.left + u * GOAL.width, GOAL.top + v * GOAL.height
    end_size = AT_GOAL * 0.9
    for i in range(n):
        t = i / (n - 1)
        x[i], y[i] = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        d[i] = start_size + t * (end_size - start_size)
    for j in range(back):
        x[n + j], y[n + j], d[n + j] = x1, y1 + 5 * j, end_size * (1 + 0.15 * (j + 1))
    return x, y, d


def judged(u, v, **kw):
    x, y, d = flight(u, v, **kw)
    shot = shots_mod.Shot(number=1, start=0, end=0, release=0)
    shots_mod.judge(shot, x, y, d, GOAL, FPS, until=len(x), ball_at_goal_px=AT_GOAL,
                    at_goal_size=1.25, post_margin=0.04, max_gap=6)
    return shot


def test_top_corner_goal():
    s = judged(0.1, 0.1)
    assert s.goal is True and s.zone == "top left", (s.goal, s.reason, s.zone, s.spot)


def test_low_right_goal():
    s = judged(0.85, 0.85)
    assert s.goal is True and s.zone == "low right", (s.reason, s.spot)


def test_wide_and_over():
    assert judged(-0.2, 0.5).reason == "wide left"
    assert judged(1.2, 0.5).reason == "wide right"
    assert judged(0.5, -0.3).reason == "over the bar"


def test_off_the_post():
    s = judged(0.99, 0.5, back=8)
    assert s.goal is False and s.reason == "off the post", s.reason


def test_saved_short_of_goal():
    x, y, d = flight(0.5, 0.5, n=20)
    d[:20] = np.linspace(160, 90, 20)          # never gets small enough...
    d[20:30] = np.linspace(100, 200, 10)       # ...then grows back: stopped
    x[20:30], y[20:30] = x[19], y[19]
    shot = shots_mod.Shot(number=1, start=0, end=0, release=0)
    shots_mod.judge(shot, x, y, d, GOAL, FPS, until=len(x), ball_at_goal_px=AT_GOAL,
                    at_goal_size=1.25, post_margin=0.04, max_gap=6)
    assert shot.goal is False and shot.reason == "saved or blocked", shot.reason


def test_no_ball():
    x = np.full(30, np.nan)
    shot = shots_mod.Shot(number=1, start=0, end=0, release=0)
    shots_mod.judge(shot, x, x, x, GOAL, FPS, until=30, ball_at_goal_px=AT_GOAL,
                    at_goal_size=1.25, post_margin=0.04, max_gap=6)
    assert shot.goal is None


def test_shot_found_in_pose():
    n = 120
    kp = np.full((n, 17, 2), np.nan)
    kp[:, :, 0] = 900
    kp[:, 0, 1] = 400                          # nose
    kp[:, 5, 1] = kp[:, 6, 1] = 480            # shoulders
    kp[:, 9, 1] = kp[:, 10, 1] = 600           # wrists low
    kp[40:50, 10, 1] = 350                     # right wrist over the head
    kp[90:100, 10, 1] = 350
    spans = shots_mod.candidates(kp, FPS, min_gap_seconds=1.0)
    assert len(spans) == 2, spans
    assert shots_mod.throwing_side(kp, spans) == "right"


def test_render_smoke(tmp_path):
    """The video and plots render end to end on a synthetic clip."""
    w, h, n = 1920, 1080, 45
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for _ in range(n):
        writer.write(np.full((h, w, 3), 60, np.uint8))
    writer.release()
    info = types.SimpleNamespace(path=clip, width=w, height=h, fps=FPS, n_frames=n)
    x, y, d = flight(0.2, 0.2, n=n - 10)
    x, y, d = x[:n], y[:n], d[:n]
    track = ball_mod.BallTrack(x, y, d, np.ones(n, bool), [(0, n)], 120.0)
    shot = shots_mod.Shot(number=1, start=0, end=2, release=0)
    shots_mod.judge(shot, x, y, d, GOAL, FPS, until=n, ball_at_goal_px=AT_GOAL,
                    at_goal_size=1.25, post_margin=0.04, max_gap=6)
    import config as cfg

    kp = np.full((n, 17, 2), np.nan)
    stats = render.render(info, kp=kp, goal=GOAL, track=track, shots=[shot],
                          dst=tmp_path / "out.mp4", cfg=cfg, console=None)
    assert stats["ball_frames_drawn"] > 0
    render.plot_summary([shot], GOAL, np.zeros((h, w, 3), np.uint8), tmp_path / "plots.png", cfg=cfg)
    render.write_csv([shot], FPS, tmp_path / "shots.csv")
    assert (tmp_path / "plots.png").stat().st_size > 0
