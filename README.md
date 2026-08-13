# SO-101 Gamepad Teleop + Recording Pipeline

Companion scripts for [LeRobot](https://github.com/huggingface/lerobot) that let you
teleoperate an SO-101 (Feetech STS3215 servos) with a gamepad, record an
imitation-learning dataset with it, and train an ACT policy on the result.
Confirmed working with a PS5 DualSense and a wired Logitech G F310 so far.

This is **not** a fork of lerobot and does not include any recorded dataset or trained
weights. A vision-based policy is tied fairly tightly to the exact camera position and
workspace it was trained on, so the point of sharing this is the *pipeline*, not a
drop-in model — you record your own dataset on your own arm/camera and train from
scratch. Expect similar results, not identical ones.

## What's in here

- `scripts/gamepad_debug.py` — prints live axis/button indices from your gamepad so
  you can confirm the mapping before touching the robot. **Run this first.**
- `scripts/gamepad_control.py` — shared axis/button constants and joint mapping, used
  by both scripts below. Edit this file to match your gamepad and preferences. The
  DualSense and Logitech G F310 both happen to report axes 0-5 and face buttons 0-3
  (Cross/A, Circle/B, Square/X, Triangle/Y) identically, so the same constants work
  for both unchanged. The D-pad is the one thing that differs by controller (buttons
  vs. a hat), and that's auto-detected at runtime — see `get_dpad_delta()`.
- `scripts/gamepad_teleop.py` — live gamepad control of the arm, no recording.
- `scripts/gamepad_record.py` — records an episodic dataset while you drive the arm
  with the gamepad (N/R/D/Q on the keyboard control episode boundaries).
- `scripts/kinesthetic_record.py` — the alternative approach we started with: disable
  torque and physically move the arm by hand while an overhead camera records. Kept
  here because it still works and some people may prefer it, but keep your hands out
  of the camera's view — hand occlusion during demonstrations was the single biggest
  cause of failed policies during development of this pipeline.
- `patches/` — two small changes to lerobot core files, described below.
- `requirements.txt` — exact pinned versions of pygame/opencv-python/torch this
  pipeline was built and tested against.

## 1. Base install

This pipeline was built and tested against lerobot at a specific commit. Later
versions of lerobot will very likely still work, but if something breaks, pin to
this exact commit first to rule out an upstream change before debugging further:

```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 3dd19d043e2f3fe5673b13ea0ebe4f31884c0797
pip install -e ".[feetech]"
pip install -r /path/to/so101_gamepad_pipeline/requirements.txt
```

If you don't have an NVIDIA GPU, drop the `--extra-index-url` line at the top of
`requirements.txt` and just let `torch` install its default (CPU-only) build —
it'll work, just slower for training/inference.

Follow lerobot's own SO-101 assembly/calibration docs first — this pipeline assumes
you already have a working, calibrated `so101_follower` that responds to
`lerobot-calibrate` / `lerobot-teleoperate` with a leader arm or the standard configs.

If you're on Windows, also read the "Windows gotchas" section below before training.

## 2. Apply the two patches

- `patches/so_follower_p_gain.patch` — lowers the servo position-loop P gain
  (32 → 16) to reduce shakiness. Apply to
  `src/lerobot/robots/so_follower/so_follower.py`.
- `patches/train_utils_exfat_symlink_fallback.patch` — only needed if your training
  output directory lives on an exFAT-formatted drive (common for external drives on
  Windows), which doesn't support symlinks. Apply to
  `src/lerobot/common/train_utils.py`.

Apply with `git apply patches/<name>.patch` from the lerobot repo root, or just make
the one-line edit by hand — the patch files describe exactly what changed and why.

## 3. Copy in the scripts

Copy everything in `scripts/` into `src/lerobot/scripts/` in your lerobot checkout.

## 4. Calibrate your gamepad mapping

```
python src/lerobot/scripts/gamepad_debug.py
```

Move one stick/trigger/button at a time and note which axis/button index changes.
Update the constants at the top of `scripts/gamepad_control.py` to match if they
differ — the indices in this repo were empirically confirmed for a PS5 DualSense and
a wired Logitech G F310, both on one specific machine, and are **not guaranteed to
match yours** (different OS/driver/USB vs Bluetooth can all shift them). If your
controller reports the D-pad as a hat rather than buttons, no changes are needed —
that's auto-detected. Also flip any `sign` values in `JOINT_CONFIG` /
`WRIST_FLEX_SIGN` / `GRIPPER_SIGN` if a control moves the wrong direction once you
test it live.

Try `gamepad_teleop.py` first to confirm every axis/button does what you expect
before recording anything.

## 5. Record a dataset

Edit the config constants at the top of `gamepad_record.py` (`PORT`, `CAMERA_INDEX`,
`NUM_EPISODES`, `SINGLE_TASK`, `REPO_ID`), then:

```
python src/lerobot/scripts/gamepad_record.py
```

Cross toggles arm motion on/off, Triangle returns to the start position (best used
between episodes, not mid-episode). N/R/D/Q on the keyboard control episode
save/re-record/delete/quit, same as lerobot's own recording scripts.

We recorded 30 episodes for our own dataset; that was enough for a working policy but
more is generally better.

## 6. Train an ACT policy

```
python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=<your_repo_id> \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --steps=6600 \
  --save_freq=2000 \
  --num_workers=4 \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name>
```

6600 steps at batch size 16 took about 40 minutes on an RTX-class GPU with 16GB VRAM
and got loss down to ~0.18 on a 30-episode/~29K-frame dataset. Scale `steps` and
`batch_size` to your dataset size and GPU. `--dataset.video_backend=pyav` is required
on Windows since torchcodec isn't available there; drop it on Linux/macOS if you have
torchcodec/FFmpeg working.

## 7. Test it

```
python -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --policy.path=outputs/train/<job_name>/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=<your_port> \
  --robot.cameras="{top: {type: opencv, index_or_path: <your_camera_index>, width: 640, height: 480, fps: 30}}" \
  --task="<your task description>" \
  --duration=60
```

Close other heavy background apps (browsers, Creative Cloud, etc.) before running —
a slow/contended CPU can drag the control loop well below its target FPS even when
the GPU has plenty of headroom, and that shows up as `Record loop is running slower
than target FPS` warnings and jerky robot motion.

## Windows gotchas

- `pip install opencv-python`, not `opencv-python-headless` — the headless build
  breaks camera preview windows, and having both installed causes import conflicts.
  Uninstall headless if you have it.
- torchcodec isn't available on Windows; always pass `--dataset.video_backend=pyav`.
- Install a CUDA build of torch explicitly if you have an NVIDIA GPU, e.g.:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` — the
  default `pip install torch` can silently give you a CPU-only build.
- exFAT-formatted drives (common for portable/external SSDs) don't support symlinks,
  which breaks lerobot's `checkpoints/last` link — see the `train_utils` patch above.
