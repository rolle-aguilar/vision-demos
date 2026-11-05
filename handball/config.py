"""Settings for the handball run. Edit here, or override per run with flags (`python main.py -h`)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT = DATA_DIR / "input"   # a clip, or a folder of clips
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion ───────────────────────────────────────────────────────────────
INFERENCE_HEIGHT = 1080   # what the models see
EXPORT_HEIGHT = 1080      # the output's total height: the clip on top, the goal chart below
TRIM_SECONDS = None       # e.g. 10.0 for a quick test
CONVERT_CRF = 23
FORCE_RECONVERT = False

# iPhone HDR looks washed out in OpenCV. "auto" tone-maps HDR clips, "off" never, "force" always.
TONEMAP = "auto"
TONEMAP_ALGORITHM = "bt.2446a"

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0
REUSE_CACHE = True        # reuse a cached reply when the request matches

# ── Pose: ViTPose ────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
POSE_FPS = 15.0
PRECISION = 4
MAX_PERSONS = 1
PERSON_IOU_THRESHOLD = 0.6

LINE_THICKNESS = 3        # px at 720 px wide; scales with the output
POINT_RADIUS = 2

SKELETON_COLORS = {       # BGR
    "left_arm":  (238, 211, 34),
    "right_arm": (241, 102, 99),
    "left_leg":  (249, 121, 232),
    "right_leg": (133, 113, 251),
    "torso":     (225, 213, 203),
}

# ── Goal: SAM 3.1, one frame ─────────────────────────────────────────────────
# The phone stays still, so the goal is found once.
SAM_MODEL = "facebook/sam3.1"
GOAL_PROMPT = "handball goal"
GOAL_FRAME_SECONDS = None   # None = the middle of the clip
GOAL_WIDTH_M = 3.0          # inside the posts (IHF)
GOAL_HEIGHT_M = 2.0

# ── Ball: SAM 3.1, around each shot only ─────────────────────────────────────
BALL_PROMPT = "ball"
BALL_BEFORE_SECONDS = 0.6     # window start, before the throwing arm goes up
BALL_AFTER_SECONDS = 1.8      # window end: a shot reaches the goal well within this
BALL_STRIDE = 1
BALL_MIN_SCORE = 0.5
BALL_SIZE_TOLERANCE = 0.6     # loose: the ball shrinks a lot on its way to the goal
BALL_DRAW_SIZE_TOLERANCE = 0.8
BALL_DIAMETER_M = 0.185       # size 3 (men); size 2 is ~0.175, size 1 ~0.16

# ── Shots ────────────────────────────────────────────────────────────────────
# A shot: the throwing wrist over the head, after both wrists were below the shoulders.
SHOT_MIN_GAP_SECONDS = 1.5    # closer than this is one shot
THROWING_HAND = "auto"        # "auto" (the wrist that goes highest), "left" or "right"

# ── Goal or miss (see how-it-works.md) ───────────────────────────────────────
AT_GOAL_SIZE = 1.25           # a ball within this × its size at the goal line is at the goal
POST_MARGIN = 0.04            # share of the goal's width near a post or the bar that counts as the frame
MAX_GAP_SECONDS = 0.2         # a path may skip this long between sightings

# ── Video ────────────────────────────────────────────────────────────────────
VERDICT_SECONDS = 1.2         # how long the goal stays lit green or red
BALL_TRAIL_SECONDS = 0.4
BALL_FILL_SECONDS = 0.2
PANEL_FONT = "auto"           # "auto", "opencv", or a font path
PANEL_FONT_INDEX = None
OUTPUT_CRF = 20

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
