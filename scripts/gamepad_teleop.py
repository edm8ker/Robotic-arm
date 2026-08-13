"""
Gamepad Teleop — drives the real SO-101 arm. Tested with a PS5 DualSense and an
Xbox-layout Logitech G F310; see gamepad_control.py for mapping details.

Velocity control: stick/trigger deflection sets how FAST a joint moves, not
a target position, since a joystick naturally centers at zero -- no neutral
calibration needed. The robot's position at the moment you press START
becomes the starting point; everything after is relative motion.

Mapping (see gamepad_control.py for exact axis/button indices and tuning):
  Left stick  X/Y  -> shoulder_pan / shoulder_lift
  Right stick X/Y  -> wrist_roll / elbow_flex
  D-pad up/down    -> wrist_flex
  L2/R2            -> gripper (R2 closes, L2 opens)

Controls:
  Cross (button 0) - toggle start/pause (robot holds position while paused)
  Triangle (button 3) - return to start position
  Circle (button 1) - quit

Usage:
  python gamepad_teleop.py
"""

import time

import pygame

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.scripts.gamepad_control import (
    BUTTON_HOME,
    BUTTON_QUIT,
    BUTTON_START_PAUSE,
    FPS,
    HOME_MOVE_DURATION_S,
    PORT,
    auto_detect_port,
    move_to_home,
    step_targets,
)


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERROR: no gamepad detected.")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Connected: {js.get_name()}")

    port = auto_detect_port(fallback=PORT)
    robot_config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(robot_config)
    print(f"Connecting to robot on {port}...")
    robot.connect()
    print("Robot connected.\n")

    home_position = dict(robot.bus.sync_read("Present_Position"))
    print("Start position captured -- Triangle will return the arm here.\n")

    print("=" * 60)
    print("  Left stick  X/Y  -> shoulder_pan / shoulder_lift")
    print("  Right stick X/Y  -> wrist_roll / elbow_flex")
    print("  D-pad up/down    -> wrist_flex")
    print("  L2/R2            -> gripper (R2 closes, L2 opens)")
    print("  Cross (button 0) - toggle start/pause")
    print("  Triangle (button 3) - return to start position")
    print("  Circle (button 1) - quit")
    print("=" * 60)
    print("\nRobot will NOT respond to the sticks until you press Cross.\n")

    running = False  # gated by Cross button
    current_targets = None
    control_interval = 1.0 / FPS

    try:
        while True:
            pygame.event.pump()

            if js.get_button(BUTTON_QUIT):
                print("\nCircle pressed -- quitting.")
                break

            if js.get_button(BUTTON_START_PAUSE):
                # Simple debounce: wait for release before toggling again.
                while js.get_button(BUTTON_START_PAUSE):
                    pygame.event.pump()
                    time.sleep(0.01)
                running = not running
                if running:
                    current_targets = dict(robot.bus.sync_read("Present_Position"))
                    print("Started -- robot responding to controller.")
                else:
                    print("Paused -- robot holding position.")

            if js.get_button(BUTTON_HOME):
                while js.get_button(BUTTON_HOME):
                    pygame.event.pump()
                    time.sleep(0.01)
                if running and current_targets is not None:
                    print("Triangle pressed -- returning to start position...")
                    current_targets = move_to_home(
                        robot, current_targets, home_position, FPS, HOME_MOVE_DURATION_S
                    )
                    print("At start position.")

            if running and current_targets is not None:
                step_targets(js, current_targets, control_interval)
                robot.bus.sync_write("Goal_Position", current_targets)

            time.sleep(control_interval)
    finally:
        if robot.is_connected:
            robot.disconnect()
        pygame.joystick.quit()
        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
