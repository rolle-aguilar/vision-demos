"""Shots: found in the pose, then judged by where the ball reaches the goal line."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .goal import Goal
from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}
HEAD = [J[n] for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]

ROWS = ("top", "middle", "low")
COLS = ("left", "centre", "right")


# ── pose, as arrays ──────────────────────────────────────────────────────────

def keypoints(by_index: dict, n_frames: int, width: int, height: int) -> np.ndarray:
    """``(n_frames, 17, 2)`` in pixels; NaN where nobody is posed or a joint is missing."""
    out = np.full((n_frames, len(KPT_NAMES), 2), np.nan)
    for i in range(n_frames):
        persons = by_index.get(i)
        if not persons:
            continue
        kpts = np.asarray(persons[0].get("kpts_xy") or [], float)
        if kpts.shape != (len(KPT_NAMES), 2):
            continue
        missing = (kpts == 0).all(axis=1)   # (0, 0) is the model's "not visible"
        kpts = kpts * [width, height]
        kpts[missing] = np.nan
        out[i] = kpts
    return out


def _median(values: np.ndarray, frames: int) -> np.ndarray:
    """Running median over *frames*, ignoring NaN."""
    half = frames // 2
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window)
    return out


# ── finding shots ────────────────────────────────────────────────────────────

def candidates(kp: np.ndarray, fps: float, *, min_gap_seconds: float) -> list[tuple[int, int]]:
    """``[(first, last frame with a wrist over the head), ...]``, one per shot.

    A handball throw is one-handed, so one wrist over the head is enough. Both wrists
    must drop below the shoulders between shots, so a follow-through can't count twice.
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        head_top = np.nanmin(kp[:, HEAD, 1], axis=1)
        high_wrist = np.nanmin(kp[:, [J["left_wrist"], J["right_wrist"]], 1], axis=1)
        low_wrist = np.nanmax(kp[:, [J["left_wrist"], J["right_wrist"]], 1], axis=1)
        shoulders = np.nanmax(kp[:, [J["left_shoulder"], J["right_shoulder"]], 1], axis=1)
        up = _median((high_wrist < head_top).astype(float), 3) > 0.5
        down = _median((low_wrist > shoulders).astype(float), 5) > 0.5

    spans: list[list[int]] = []
    armed = True
    i = 0
    while i < len(up):
        if armed and up[i]:
            j = i
            while j + 1 < len(up) and up[j + 1]:
                j += 1
            if spans and i - spans[-1][1] < min_gap_seconds * fps:
                spans[-1][1] = j
            else:
                spans.append([i, j])
            armed = False
            i = j + 1
            continue
        if down[i]:
            armed = True
        i += 1
    return [(a, b) for a, b in spans]


def throwing_side(kp: np.ndarray, spans: list[tuple[int, int]]) -> str:
    """The arm whose wrist is higher while the shots are overhead."""
    height = {}
    for side in ("left", "right"):
        ys = [kp[a:b + 1, J[f"{side}_wrist"], 1] for a, b in spans]
        ys = np.concatenate(ys) if ys else np.array([])
        height[side] = -np.nanmedian(ys) if np.isfinite(ys).any() else -np.inf
    return max(height, key=height.get)


def find_release(start: int, end: int, bx: np.ndarray, by: np.ndarray, wrist: np.ndarray,
                 diameter: float, fps: float) -> int:
    """The first frame the ball is 2.5 ball sizes from the throwing wrist and pulling away.

    Without a ball, the end of the overhead span.
    """
    lo, hi = max(0, start - int(0.2 * fps)), min(len(bx) - 3, end + int(0.6 * fps))
    gap = np.hypot(bx - wrist[:, 0], by - wrist[:, 1])
    for f in range(lo, hi):
        nxt = gap[f + 1:f + 3]
        if (np.isfinite(gap[f]) and gap[f] > 2.5 * diameter
                and np.isfinite(nxt).all() and nxt[0] > gap[f] and nxt[1] > nxt[0]):
            return f
    return end


# ── one shot ─────────────────────────────────────────────────────────────────

@dataclass
class Shot:
    number: int
    start: int                 # first frame with the throwing hand overhead
    end: int                   # last frame overhead
    release: int               # the ball leaves the hand
    goal: bool | None = None   # None: the ball was never followed to the goal
    reason: str = ""
    at_goal: int | None = None      # frame the ball reached the goal line
    decided: int | None = None      # frame the verdict shows on the video
    spot: tuple[float, float] | None = None   # on the goal face: 0..1 across, 0..1 down
    path: list[tuple[int, float, float]] = field(default_factory=list)   # ball (frame, x, y)

    @property
    def label(self) -> str:
        return "GOAL" if self.goal else "MISS" if self.goal is False else "?"

    @property
    def zone(self) -> str | None:
        """``"top left"``, ``"low centre"``, ... for a spot inside the goal."""
        if self.spot is None or not (0 <= self.spot[0] <= 1 and 0 <= self.spot[1] <= 1):
            return None
        col = COLS[min(2, int(self.spot[0] * 3))]
        row = ROWS[min(2, int(self.spot[1] * 3))]
        return f"{row} {col}"

    def summary(self, fps: float) -> dict:
        return {
            "shot": self.number, "result": self.label.lower(), "reason": self.reason,
            "release_s": round(self.release / fps, 2),
            "at_goal_s": round(self.at_goal / fps, 2) if self.at_goal is not None else None,
            "flight_s": round((self.at_goal - self.release) / fps, 2)
            if self.at_goal is not None else None,
            "zone": self.zone,
            "spot": [round(v, 3) for v in self.spot] if self.spot else None,
        }


def judge(shot: Shot, x: np.ndarray, y: np.ndarray, size: np.ndarray, goal: Goal, fps: float, *,
          until: int, ball_at_goal_px: float, at_goal_size: float, post_margin: float,
          max_gap: int) -> None:
    """Goal or miss, and where.

    Filmed from behind the shooter, the ball shrinks as it flies away. It is at the goal
    line once it shrinks to about its size there (the goal's 3 m width gives the scale).
    Where it is in the image at that moment is where it crossed the line: inside the
    posts and under the bar is a goal; near the frame, it's a goal only if the ball
    doesn't come back off the post or bar; anywhere else is a miss. A ball that grows
    again before it ever gets that small was stopped short: saved or blocked.
    """
    frames = [f for f in range(shot.release, min(until, len(x)))
              if np.isfinite(x[f]) and np.isfinite(size[f])]
    shot.path = [(f, float(x[f]), float(y[f])) for f in frames]
    if len(frames) < 3:
        shot.goal, shot.reason = None, "ball not tracked"
        return

    # A short path: stop at the first gap too long to bridge.
    kept = [frames[0]]
    for f in frames[1:]:
        if f - kept[-1] > max_gap:
            break
        kept.append(f)
    frames = kept
    s = _median(np.array([size[f] for f in frames]), 3)

    limit = at_goal_size * ball_at_goal_px
    hit = next((k for k in range(len(frames)) if s[k] <= limit), None)
    if hit is None:
        smallest = int(np.argmin(s))
        grew = smallest < len(frames) - 1 and np.max(s[smallest:]) > 1.3 * s[smallest]
        shot.decided = frames[smallest] if grew else frames[-1]
        shot.goal = False if grew else None
        shot.reason = "saved or blocked" if grew else "lost before the goal"
        return

    # From there it is nearly at the line: follow it while it keeps shrinking (for a
    # moment), and take its farthest point as where it crossed.
    window = [k for k in range(hit, len(frames)) if frames[k] - frames[hit] <= 0.25 * fps]
    hit = window[int(np.argmin([s[k] for k in window]))]
    f = frames[hit]
    px, py = x[f], y[f]
    u, v = goal.face(px, py)
    shot.at_goal, shot.spot = f, (float(u), float(v))
    mx = post_margin
    my = post_margin * goal.width / goal.height

    # After the goal line, does the ball come back towards the camera (off the frame)?
    after = s[hit:]
    rebound = len(after) > 2 and np.max(after) > 1.5 * np.min(after[:3])

    if v < -my:
        shot.goal, shot.reason = False, "over the bar"
    elif u < -mx:
        shot.goal, shot.reason = False, "wide left"
    elif u > 1 + mx:
        shot.goal, shot.reason = False, "wide right"
    elif v > 1 + my:
        # Under the goal line: it bounced in front of the goal. Count it if it's between the posts.
        inside = 0 <= u <= 1
        shot.goal = inside and not rebound
        shot.reason = "bounce shot" if shot.goal else "bounced out"
    elif u < mx or u > 1 - mx or v < my:
        where = "post" if (u < mx or u > 1 - mx) else "bar"
        shot.goal = not rebound
        shot.reason = f"in off the {where}" if shot.goal else f"off the {where}"
    else:
        shot.goal, shot.reason = not rebound, "in the net" if not rebound else "saved"
    shot.decided = f + max(1, int(0.15 * fps))


def ball_at_goal_px(goal: Goal, *, ball_m: float, goal_width_m: float) -> float:
    """How wide the ball looks at the goal line, px."""
    return ball_m / goal_width_m * goal.width
