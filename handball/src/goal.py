"""The goal: SAM 3.1 on one frame, then the goal mouth read off the mask.

The phone stays still and faces the goal, so the posts and crossbar make a
rectangle in the image. Its width is 3 m and its height 2 m, which also gives the
ball's size at the goal line.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

CONTENT_OBJECT = "img.segment.masks"
CONTRACT = "sam31-goal-v1"


@dataclass(frozen=True)
class Goal:
    """The goal mouth in pixels of the inference frame."""

    left: float        # inside of the left post (as seen)
    right: float
    top: float         # underside of the crossbar
    bottom: float      # the goal line on the floor
    score: float
    frame: int         # the frame SAM looked at

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}

    def face(self, x: float, y: float) -> tuple[float, float]:
        """*(x, y)* px as a spot on the goal face: 0..1 left to right, 0..1 top to bottom."""
        return (x - self.left) / self.width, (y - self.top) / self.height



def request_segment(client, *, model: str, image_png: bytes, prompt: str) -> tuple[dict, dict | None]:
    """Every instance of *prompt* in one image, with a label map."""
    b64 = base64.b64encode(image_png).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        response_format={"type": "json_object"},
        extra_body={"method": "segment", "method_params": {"prompt": prompt, "mask_format": "png"}},
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def read_frame(mp4: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {mp4}")
    return frame


def _label_map(mask) -> np.ndarray | None:
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    raw = base64.b64decode(str(mask.get("data", "")).split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    return labels[:, :, 0] if labels.ndim == 3 else labels

def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str, frame: int) -> Path:
    stat = video.stat()
    key = json.dumps({"video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "model": model, "prompt": prompt, "frame": frame, "contract": CONTRACT},
                     sort_keys=True)
    return cache_dir / f"goal.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def locate(payload: dict, *, frame: int, width: int, height: int) -> Goal:
    """The best-scoring goal. Its mask's box is the mouth: from the front the net sits
    inside the frame, so the posts and bar are the mask's outer edges."""
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got "
                           f"{json.dumps(payload)[:300]}")
    items = content.get("items") or []
    if not items:
        raise RuntimeError("SAM found no goal in the frame. Is the whole goal in shot? "
                           "Try another GOAL_FRAME_SECONDS.")
    best = max(items, key=lambda i: float(i.get("score") or 0))
    bx, by, bw, bh = best["bbox_xywh"]
    left, right, top, bottom = bx * width, (bx + bw) * width, by * height, (by + bh) * height

    labels = _label_map(content.get("mask"))
    instance = best.get("instance_id")
    if labels is not None and instance is not None:
        blob = labels == int(instance)
        rows = np.nonzero(blob.any(axis=1))[0]
        cols = np.nonzero(blob.any(axis=0))[0]
        if rows.size and cols.size:
            top, bottom = float(rows.min()), float(rows.max() + 1)
            left, right = float(cols.min()), float(cols.max() + 1)
    return Goal(left=left, right=right, top=top, bottom=bottom,
                score=float(best.get("score") or 0), frame=frame)
