# handball

Tracks a handball shooting session from a phone that stays still. It calls every
shot a **goal or a miss** by following the ball to the goal line, says **which
part of the goal** it went in (top left, low centre, ...), and draws every shot on
a front-on goal with a running goal count.

Pose runs on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large),
the goal and ball on [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1),
both through the [VLM Run Gateway](https://www.vlm.run/gateway). How it works:
[how-it-works.md](how-it-works.md).

Built from the [basketball](../basketball/) demo.

## Filming

- Phone set down and still, **landscape**, behind the shooter and a little to one side.
- The **whole goal** and the shooter in frame, the phone at about chest height and
  square to the goal, so the posts and crossbar look like a rectangle.
- Good light; 60 fps if the phone has it. 5 to 10 shots per clip.
- No goalkeeper for now: a save right on the line can read as a goal.

## Run it

1. Get an API key at [app.vlm.run](https://app.vlm.run/sign-in) and put it in a `.env` at the repo root:

   ```bash
   cp ../.env.example ../.env   # then paste your key after VLMRUN_API_KEY=
   ```

2. Create the environment (or `pip install -r ../requirements.txt`; ffmpeg is needed too):

   ```bash
   conda env create -f environment.yml
   conda activate handball
   ```

3. Put a clip in `data/input/`.

4. Run it:

   ```bash
   python main.py                         # every clip in data/input/
   python main.py data/input/session.mov  # one clip
   ```

   | flag | sets |
   |---|---|
   | `--hand right` | the throwing hand (default: auto) |
   | `--ball 0.175` | the ball's diameter in meters: 0.185 size 3, 0.175 size 2, 0.16 size 1 |
   | `--height 1080` | the output's total height; the clip and the scoreboard below each take half |
   | `--trim 10` | only the first 10 s |
   | `--fresh` | ignore cached replies |

   Everything else is in [`config.py`](config.py).

## Output

Each run writes to `data/output/<timestamp>/`:

| file | what |
|---|---|
| `<clip>_handball.mp4` | the video on top; goals, the last shot and the goal chart below |
| `plots.png` | every ball path over the frame, and the goal chart with goals per zone |
| `<clip>_shots.csv` | one row per shot: result, why, zone, flight time |
| `run.json` | settings, per-shot results and cost |

Model replies are cached in `data/cache/`, so re-runs don't call the models again.

## Tests

`python -m pytest tests` checks the goal-or-miss logic and the render on made-up
ball paths, with no gateway calls.

## License

[Apache-2.0](../LICENSE).
