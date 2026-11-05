"""Track a handball shooting session: every shot, goal or miss, and where it went in.

    python main.py                         # every clip in data/input/
    python main.py data/input/session.mov  # one clip
    python main.py -h                      # every flag

ViTPose finds the shots; SAM 3.1 finds the goal, then tracks the ball around each shot.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import ball as ball_mod
from src import goal as goal_mod, pose, render, shots as shots_mod, video
from src.env import load_api_key

console = Console()


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def kv(pairs: dict, title: str | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, title_align="left", expand=False) if title else table)


def rel(path: Path) -> Path:
    """*path* relative to the project, for printing."""
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def stamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                  timeout=cfg.REQUEST_TIMEOUT, max_retries=1)


def cost_of(usage: dict | None) -> float:
    return float((usage or {}).get("cost") or 0)


# ── models ───────────────────────────────────────────────────────────────────

def pose_clip(api_key: str, mp4: Path, info: video.VideoInfo) -> dict:
    """ViTPose over the whole clip, cached."""
    video_b64, extra_body = pose.build_request(mp4, video_fps=cfg.POSE_FPS,
                                               max_frames=info.n_frames, precision=cfg.PRECISION)
    cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL, extra_body=extra_body)
    cached = pose.load_cache(cache) if cfg.REUSE_CACHE else None
    seconds = None
    if cached is not None:
        payload, usage, created = cached
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/]; no gateway call")
    else:
        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; ViTPose on every frame…",
                            spinner="dots"):
            payload, usage = pose.request_poses(client(api_key), model=cfg.POSE_MODEL,
                                                video_b64=video_b64, extra_body=extra_body)
        seconds = time.perf_counter() - t0
        pose.save_cache(cache, payload, usage, stamp=stamp_now())
        console.print(f"  [green]done in {seconds:.1f}s[/]")

    frames, _ = pose.unwrap(payload)
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    return {"by_index": {f["index"]: f["persons"] for f in frames}, "dropped": dropped,
            "seconds": seconds, "extra_body": extra_body,
            "cost": 0.0 if cached is not None else cost_of(usage)}



def find_goal(api_key: str, mp4: Path, info: video.VideoInfo) -> tuple[goal_mod.Goal, float]:
    """SAM 3.1 on one frame, cached. The phone stays still, so it holds for the clip."""
    seconds = cfg.GOAL_FRAME_SECONDS
    frame = info.n_frames // 2 if seconds is None else min(info.n_frames - 1, int(seconds * info.fps))
    cache = goal_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.SAM_MODEL, prompt=cfg.GOAL_PROMPT,
                                frame=frame)
    blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
    cost = 0.0
    if blob is None:
        ok, png = cv2.imencode(".png", goal_mod.read_frame(mp4, frame))
        with console.status("[cyan]waiting on the gateway[/]; SAM is finding the goal…",
                            spinner="dots"):
            payload, usage = goal_mod.request_segment(client(api_key), model=cfg.SAM_MODEL,
                                                      image_png=png.tobytes(), prompt=cfg.GOAL_PROMPT)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"created": stamp_now(), "payload": payload, "usage": usage}))
        cost = cost_of(usage)
    else:
        payload = blob["payload"]
        console.print(f"  [green]cache hit[/]: goal from [dim]{blob.get('created')}[/]")
    return goal_mod.locate(payload, frame=frame, width=info.width, height=info.height), cost


def track_window(api_key: str, mp4: Path, info: video.VideoInfo, start: int, length: int
                 ) -> tuple[dict, float]:
    """SAM 3.1 on frames ``[start, start + length)``, cached; detections in clip frames."""
    cache = ball_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.SAM_MODEL, prompt=cfg.BALL_PROMPT,
                                start=start, length=length, stride=cfg.BALL_STRIDE)
    blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
    cost = 0.0
    if blob is None:
        seg = cfg.CACHE_DIR / f"{mp4.stem}.win{start}_{length}.mp4"
        if not seg.is_file():
            video.segment(mp4, seg, start_frame=start, n_frames=length, fps=info.fps)
        payload, usage = ball_mod.request_track(
            client(api_key), model=cfg.SAM_MODEL,
            video_b64=base64.b64encode(seg.read_bytes()).decode("ascii"),
            prompt=cfg.BALL_PROMPT, stride=cfg.BALL_STRIDE)
        cache.write_text(json.dumps({"created": stamp_now(), "payload": payload, "usage": usage}))
        seg.unlink(missing_ok=True)
        cost = cost_of(usage)
    else:
        payload = blob["payload"]
    return ball_mod.detections(payload, frame_offset=start, width=info.width, height=info.height,
                               min_score=cfg.BALL_MIN_SCORE), cost


def track_ball(api_key: str, mp4: Path, info: video.VideoInfo, spans, min_size: float
               ) -> tuple[ball_mod.BallTrack, float, int]:
    """The ball around every shot. A throw reaches the goal fast, so one window per shot."""
    fps = info.fps
    cap = ball_mod.TRACK_MAX_FRAMES * cfg.BALL_STRIDE
    raw: dict[int, list] = {}
    windows: list[tuple[int, int]] = []
    cost, calls = 0.0, 0
    with console.status("[cyan]waiting on the gateway[/]; SAM is tracking the ball…",
                        spinner="dots") as status:
        for n, (a, _) in enumerate(spans, 1):
            start = max(0, a - int(cfg.BALL_BEFORE_SECONDS * fps))
            end = min(info.n_frames, a + int(cfg.BALL_AFTER_SECONDS * fps))
            if windows and start < sum(windows[-1]):
                start = sum(windows[-1])   # never pay twice for a frame
            while start < end:
                length = min(end - start, cap)
                status.update(f"[cyan]waiting on the gateway[/]; SAM is tracking the ball "
                              f"(shot {n}/{len(spans)})…")
                found, c = track_window(api_key, mp4, info, start, length)
                raw.update(found)
                windows.append((start, length))
                cost += c
                calls += 1
                start += length
    track = ball_mod.select(raw, info.n_frames, windows, min_size=min_size,
                            size_tolerance=cfg.BALL_SIZE_TOLERANCE)
    track.shown = ball_mod.select(raw, info.n_frames, windows, min_size=min_size, local_size=True,
                                  size_tolerance=cfg.BALL_DRAW_SIZE_TOLERANCE)
    return track, cost, calls


# ── the pipeline ─────────────────────────────────────────────────────────────

def analyze(kp: np.ndarray, info: video.VideoInfo, spans, goal, track, side: str,
            at_goal_px: float) -> list[shots_mod.Shot]:
    fps = info.fps
    wrist = kp[:, shots_mod.J[f"{side}_wrist"], :2]
    shots = []
    for n, (a, b) in enumerate(spans, 1):
        nxt = spans[n][0] if n < len(spans) else info.n_frames
        release = shots_mod.find_release(a, b, track.x, track.y, wrist, track.diameter, fps)
        shot = shots_mod.Shot(number=n, start=a, end=b, release=release)
        covered_until = release
        while covered_until < info.n_frames and track.covered[covered_until]:
            covered_until += 1
        shots_mod.judge(shot, track.x, track.y, track.d, goal, fps, until=min(nxt, covered_until),
                        ball_at_goal_px=at_goal_px, at_goal_size=cfg.AT_GOAL_SIZE,
                        post_margin=cfg.POST_MARGIN, max_gap=max(2, int(cfg.MAX_GAP_SECONDS * fps)))
        shots.append(shot)
    return shots


def run_clip(api_key: str) -> dict:
    """The whole pipeline for ``cfg.INPUT_VIDEO``. Returns what the batch summary prints."""
    t_start = time.perf_counter()

    rule(2, "Source video")
    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({
        "path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
    })

    rule(3, "Convert to MP4")
    tonemap, reason = video.tonemap_filter(src_info, mode=cfg.TONEMAP,
                                           algorithm=cfg.TONEMAP_ALGORITHM)
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{reason}")
    mp4 = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=cfg.INFERENCE_HEIGHT,
                           trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF, tonemap=tonemap)
    if mp4.is_file() and not cfg.FORCE_RECONVERT:
        console.print(f"  [green]cache hit[/]: [dim]{rel(mp4)}[/]")
    else:
        video.convert(cfg.INPUT_VIDEO, mp4, target_height=cfg.INFERENCE_HEIGHT,
                      trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                      total_frames=src_info["n_frames"], console=console, tonemap=tonemap)
    info = video.inspect(mp4)
    console.print(f"  [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")

    rule(4, "Pose (ViTPose)")
    posed = pose_clip(api_key, mp4, info)
    kp = shots_mod.keypoints(posed["by_index"], info.n_frames, info.width, info.height)
    n_posed = int(np.isfinite(kp[:, 5, 0]).sum())
    spans = shots_mod.candidates(kp, info.fps, min_gap_seconds=cfg.SHOT_MIN_GAP_SECONDS)
    side = (cfg.THROWING_HAND if cfg.THROWING_HAND in ("left", "right")
            else shots_mod.throwing_side(kp, spans))
    kv({"model": cfg.POSE_MODEL, "frames with a person": f"{n_posed} of {info.n_frames}",
        "shots found": f"{len(spans)}, at " + ", ".join(f"{a / info.fps:.1f}s" for a, _ in spans),
        "throwing hand": side + ("" if cfg.THROWING_HAND in ("left", "right") else " (auto)"),
        "cost": f"${posed['cost']:.4f}"}, title="pose")

    rule(5, "Goal and ball (SAM 3.1)")
    goal, goal_cost = find_goal(api_key, mp4, info)
    at_goal_px = shots_mod.ball_at_goal_px(goal, ball_m=cfg.BALL_DIAMETER_M,
                                           goal_width_m=cfg.GOAL_WIDTH_M)
    track, ball_cost, calls = track_ball(api_key, mp4, info, spans, min_size=0.6 * at_goal_px)
    tracked = int(track.covered.sum())
    kv({"goal": f"{goal.width:.0f} × {goal.height:.0f} px at ({goal.left:.0f}, {goal.top:.0f}) "
                f"(score {goal.score:.2f}); the ball is ~{at_goal_px:.0f} px there",
        "ball windows": f"{calls} calls, {tracked} of {info.n_frames} frames",
        "ball found": f"{int(track.found.sum())} frames, ~{track.diameter:.0f} px across in hand",
        "cost": f"${goal_cost + ball_cost:.4f}"}, title="SAM 3.1")

    rule(6, "Shots: goal or miss")
    shots = analyze(kp, info, spans, goal, track, side, at_goal_px)
    goals = sum(1 for s in shots if s.goal)
    table = Table(box=None, pad_edge=False)
    for col in ("shot", "at", "result", "why", "zone", "flight"):
        table.add_column(col, justify="left" if col in ("result", "why", "zone") else "right",
                         style="dim" if col == "shot" else None)
    for s in shots:
        flight = f"{(s.at_goal - s.release) / info.fps:.2f}s" if s.at_goal is not None else "—"
        table.add_row(str(s.number), f"{s.release / info.fps:.1f}s",
                      "[green]GOAL[/]" if s.goal else "[red]MISS[/]" if s.goal is False else "[dim]?[/]",
                      s.reason, s.zone or "—", flight)
    console.print(Panel(table, title=f"[bold]{goals}/{len(shots)}[/] goals" if shots else "no shots",
                        title_align="left", expand=False))

    rule(7, "Render")
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / stamp
    n = 2
    while run_dir.exists():
        run_dir = cfg.OUTPUT_DIR / f"{stamp}-{n}"
        n += 1
    stamp = run_dir.name
    run_dir.mkdir(parents=True)

    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(info, kp=kp, goal=goal, track=track, shots=shots, dst=raw, cfg=cfg,
                          console=console)
    render_seconds = time.perf_counter() - t0
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_handball.mp4"
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB, {render_seconds:.1f}s)")
    plots = run_dir / "plots.png"
    render.plot_summary(shots, goal, goal_mod.read_frame(mp4, goal.frame), plots, cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(plots)}[/]")
    csv_path = run_dir / f"{cfg.INPUT_VIDEO.stem}_shots.csv"
    render.write_csv(shots, info.fps, csv_path)
    console.print(f"  data   -> [dim]{rel(csv_path)}[/]")

    total = time.perf_counter() - t_start
    cost = posed["cost"] + goal_cost + ball_cost
    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp,
        "total_seconds": round(total, 2),
        "input": {"path": str(cfg.INPUT_VIDEO), **src_info},
        "converted": {"path": str(mp4), "width": info.width, "height": info.height,
                      "fps": info.fps, "n_frames": info.n_frames, "tonemap": tonemap},
        "pose": {"model": cfg.POSE_MODEL, "extra_body": posed["extra_body"], "frames_posed": n_posed},
        "goal": {"model": cfg.SAM_MODEL, "prompt": cfg.GOAL_PROMPT, **goal.to_dict(),
                 "ball_at_goal_px": round(at_goal_px, 1)},
        "ball": {"model": cfg.SAM_MODEL, "prompt": cfg.BALL_PROMPT, "stride": cfg.BALL_STRIDE,
                 "windows": track.windows, "frames_tracked": tracked,
                 "frames_found": int(track.found.sum())},
        "throwing_hand": side,
        "shots": [s.summary(info.fps) for s in shots],
        "goals": goals, "attempts": len(shots),
        "cost_usd": {"pose": round(posed["cost"], 6), "goal": round(goal_cost, 6),
                     "ball": round(ball_cost, 6), "total": round(cost, 6)},
        "render": {**stats, "seconds": round(render_seconds, 2)},
    }, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    console.print()
    kv({"pose": f"${posed['cost']:.4f}", "goal": f"${goal_cost:.4f}", "ball": f"${ball_cost:.4f}",
        "total": f"[bold]${cost:.4f}[/]"}, title="cost")
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return {"goals": goals, "attempts": len(shots), "cost": cost, "seconds": total, "output": run_dir}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The per-run settings, as flags. Anything left out keeps its config.py value."""
    parser = argparse.ArgumentParser(
        description="Track a handball shooting session: every shot, goal or miss, and where it "
                    "went in. Flags override config.py for this run only.")
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips, or folders of clips, to run (config: {rel(cfg.INPUT)})")
    parser.add_argument("--height", type=int, metavar="PX",
                        help=f"the output's total height (config: {cfg.EXPORT_HEIGHT})")
    parser.add_argument("--trim", type=float, metavar="SECONDS",
                        help="use only the first SECONDS of each clip, for a quick test")
    parser.add_argument("--hand", choices=("auto", "left", "right"),
                        help=f"the throwing hand (config: {cfg.THROWING_HAND})")
    parser.add_argument("--ball", type=float, metavar="M",
                        help=f"the ball's diameter in meters: 0.185 size 3, 0.175 size 2, 0.16 "
                             f"size 1 (config: {cfg.BALL_DIAMETER_M:g})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore every cached gateway reply and call the models again")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Write the flags over the config, before anything reads it."""
    if args.height:
        cfg.EXPORT_HEIGHT = args.height
    if args.trim:
        cfg.TRIM_SECONDS = args.trim
    if args.hand:
        cfg.THROWING_HAND = args.hand
    if args.ball:
        cfg.BALL_DIAMETER_M = args.ball
    if args.fresh:
        cfg.REUSE_CACHE = False


def find_clips(inputs: list[Path]) -> list[Path]:
    """Every clip named, and every video directly inside every folder named, in order."""
    clips: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_dir():
            clips += sorted(p for p in path.iterdir()
                            if p.is_file() and not p.name.startswith(".")
                            and p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
        elif path.is_file():
            clips.append(path)
        else:
            console.print(f"[red]Not found:[/] {item}")
    return list(dict.fromkeys(clips))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_args(args)
    clips = find_clips(args.inputs or [cfg.INPUT])
    console.print()
    console.rule("[bold]Handball: goals, misses and where they went in[/]", align="center")
    if not clips:
        console.print(f"[red]No clips to run.[/] Put a shooting video in "
                      f"{rel(cfg.INPUT)}/, or pass one: python main.py path/to/clip.mov")
        return 1

    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  key loaded from [green]{source}[/]")

    results: dict[str, dict | str] = {}
    for n, clip in enumerate(clips, 1):
        if len(clips) > 1:
            console.print()
            console.rule(f"[bold magenta]clip {n}/{len(clips)}[/] {clip.name}", align="left")
        cfg.INPUT_VIDEO = clip
        try:
            results[clip.name] = run_clip(api_key)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console.print(f"\n[red]error on {clip.name}:[/] {exc}")
            results[clip.name] = str(exc).splitlines()[0][:100]

    if len(clips) > 1:
        table = Table(title="batch", box=None, pad_edge=False, title_justify="left")
        for col in ("clip", "goals", "cost", "output"):
            table.add_column(col)
        for name, r in results.items():
            if isinstance(r, str):
                table.add_row(name, "[red]failed[/]", r, "")
            else:
                table.add_row(name, f"{r['goals']}/{r['attempts']}", f"${r['cost']:.3f}",
                              str(rel(r["output"])))
        console.print(table)
    return 0 if all(not isinstance(r, str) for r in results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
