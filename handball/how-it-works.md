# How it works

Every threshold is in [`config.py`](config.py).

## Shots

ViTPose poses the shooter on every frame. A shot is one wrist over the head, after
both wrists were below the shoulders (a handball throw is one-handed). The
throwing hand is the wrist that goes highest.

## Goal

The phone stays still, so SAM 3.1 finds the goal once, on one frame, with the
prompt "handball goal". Seen from the front, the net sits inside the frame, so the
mask's outer edges are the posts, the crossbar and the goal line.

A handball goal is 3 m wide, which sets the scale: a 0.185 m ball is
0.185 / 3 of the goal's width in pixels when it reaches the goal line.

## Ball

SAM tracks the ball only around each shot, from 0.6 s before the arm goes up to
1.8 s after. The ball shrinks several times over on its way to the goal, so boxes
are kept when they are round and change size smoothly, not by one fixed size.

The release is the first frame the ball is 2.5 ball sizes from the throwing wrist
and pulling away.

## Goal or miss

Filmed from behind, the ball shrinks as it flies away. It is **at the goal line**
once it is within 1.25× its size there; it is then followed for up to 0.25 s while
it keeps shrinking, and its farthest point is where it crossed.

That point on the goal face decides the call:

- **Inside** the posts and under the bar: a goal, in one of nine zones (top, middle,
  low × left, centre, right).
- **Within 4%** of a post or the bar: a goal unless the ball comes back towards the
  camera afterwards ("off the post", "off the bar").
- **Outside**: a miss, "wide left", "wide right" or "over the bar".
- **Below the goal line** but between the posts: it bounced in front, a goal if it
  carries on into the net.

A ball that grows again before it ever gets that small was stopped short: "saved
or blocked". A ball lost before the goal is left uncalled ("?").

## Limits

- One camera can't tell a keeper's save right on the line from a goal; shoot at an
  empty goal for now.
- The goal must look roughly rectangular: film square to it.
