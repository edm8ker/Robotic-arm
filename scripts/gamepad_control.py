"""Shared DualSense gamepad control logic for SO-101 teleop and recording.

Pulled out of gamepad_teleop.py so gamepad_record.py can reuse the exact
same axis/button mapping and tuning (deadzones, signs, speeds) without the
two scripts drifting apart after future tuning changes.
"""

import time

PORT = "COM8"
FPS = 50
DEADZONE = 0.08

# Axis indices confirmed via gamepad_debug.py for this controller (pygame's
# real 0-indexed values -- this controller only has 6 axes, 0-5).
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_L2 = 4
AXIS_R2 = 5
BUTTON_DPAD_UP = 11
BUTTON_DPAD_DOWN = 12
BUTTON_START_PAUSE = 0  # Cross
BUTTON_QUIT = 1  # Circle
# Guessed from the standard DualSense face-button order (Cross=0, Circle=1,
# Square=2, Triangle=3) since Cross/Circle already matched that sequence --
# not yet empirically confirmed like the axes were.
BUTTON_HOME = 3  # Triangle

# How long the smooth return-to-start move takes, so pressing Triangle from
# far away doesn't yank the arm at full speed.
HOME_MOVE_DURATION_S = 1.5

# Which stick axis drives each joint, its sign, and how fast it moves
# (degrees/sec at full stick deflection).
JOINT_CONFIG = {
    "shoulder_pan":  {"axis": AXIS_LEFT_X,  "sign": 1,  "max_deg_per_s": 40.0},
    "shoulder_lift": {"axis": AXIS_LEFT_Y,  "sign": -1, "max_deg_per_s": 40.0},
    "elbow_flex":    {"axis": AXIS_RIGHT_Y, "sign": -1, "max_deg_per_s": 40.0},
    # Wrist roll gets a wider deadzone -- this axis showed small phantom
    # values (~0.05) at rest that the global deadzone didn't fully catch,
    # causing slow unintended drift.
    "wrist_roll":    {"axis": AXIS_RIGHT_X, "sign": -1, "max_deg_per_s": 60.0, "deadzone": 0.2},
}
WRIST_FLEX_SIGN = -1
WRIST_FLEX_MAX_DEG_PER_S = 40.0
GRIPPER_SIGN = -1  # positive = R2 closes, L2 opens (matches "lower = more closed" convention)
GRIPPER_MAX_PER_S = 60.0  # units/sec on the 0-100 gripper scale

# Soft position limits (degrees) so a runaway stick can't drive a joint past
# a sane range even without hitting the servo's own hard limits. Gripper is
# clamped separately to its native 0-100 scale.
JOINT_MIN_DEG = -100.0
JOINT_MAX_DEG = 100.0


def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    return 0.0 if abs(value) < deadzone else value


def step_targets(js, current_targets: dict, dt: float) -> dict:
    """Advance current_targets by one control-loop tick based on live stick/trigger state.

    Mutates and returns current_targets.
    """
    for joint, cfg in JOINT_CONFIG.items():
        raw = apply_deadzone(js.get_axis(cfg["axis"]), cfg.get("deadzone", DEADZONE))
        vel = cfg["sign"] * raw * cfg["max_deg_per_s"]
        new_val = current_targets[joint] + vel * dt
        current_targets[joint] = max(JOINT_MIN_DEG, min(JOINT_MAX_DEG, new_val))

    dpad_delta = 0.0
    if js.get_button(BUTTON_DPAD_UP):
        dpad_delta += 1.0
    if js.get_button(BUTTON_DPAD_DOWN):
        dpad_delta -= 1.0
    if dpad_delta != 0.0:
        vel = WRIST_FLEX_SIGN * dpad_delta * WRIST_FLEX_MAX_DEG_PER_S
        new_val = current_targets["wrist_flex"] + vel * dt
        current_targets["wrist_flex"] = max(JOINT_MIN_DEG, min(JOINT_MAX_DEG, new_val))

    l2 = (js.get_axis(AXIS_L2) + 1.0) / 2.0  # 0 (released) .. 1 (pressed)
    r2 = (js.get_axis(AXIS_R2) + 1.0) / 2.0
    gripper_vel = GRIPPER_SIGN * (r2 - l2) * GRIPPER_MAX_PER_S
    new_gripper = current_targets["gripper"] + gripper_vel * dt
    current_targets["gripper"] = max(0.0, min(100.0, new_gripper))

    return current_targets


def move_to_home(robot, start: dict, home_position: dict, fps: int, duration_s: float = HOME_MOVE_DURATION_S) -> dict:
    """Blocking, smooth ramp from `start` to `home_position`. Returns the final target dict."""
    steps = max(1, int(duration_s * fps))
    control_interval = 1.0 / fps
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = {joint: start[joint] + (home_position[joint] - start[joint]) * alpha for joint in start}
        robot.bus.sync_write("Goal_Position", interp)
        time.sleep(control_interval)
    return dict(home_position)
