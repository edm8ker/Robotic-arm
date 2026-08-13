#!/usr/bin/env python
"""Record a dataset via gamepad teleop (see gamepad_control.py for the stick/button
mapping and gamepad_teleop.py for a control-only version). Tested with a PS5
DualSense and an Xbox-layout Logitech G F310.

Unlike kinesthetic_record.py, the arm is never physically touched -- you
drive it with the gamepad, so your hands (and the camera's view of the
scene) stay completely clear. The commanded gamepad position is recorded as
the ACTION; the arm's actual encoder reading is recorded as the OBSERVATION
(they can differ slightly since the servo takes a moment to catch up).

Controls:
  Cross (button 0)     - toggle start/pause of arm motion (independent of
                          recording -- lets you reposition without it
                          counting as part of an episode)
  Triangle (button 3)  - return arm to the position it was in when this
                          script connected. Best used BETWEEN episodes --
                          using it mid-episode creates a gap in that
                          episode's frames since the ramp isn't recorded.
  N / right arrow - while waiting: start recording now
                    while recording: end the episode now (success)
  R / left arrow  - discard and re-record the CURRENT episode (while recording)
  D               - while waiting: mark the last SAVED episode for deletion.
                    Press again to mark the one before that, and so on.
                    Removed from the dataset in one pass at the end.
  Q / esc         - stop recording entirely (keeps episodes already saved,
                    still applies any pending deletions)

There is no auto-start between episodes -- recording only begins once you
press N, so take as long as you need to reset the scene between episodes.

Usage:
    python gamepad_record.py
"""

import shutil
import time

import pygame

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.scripts.gamepad_control import (
    BUTTON_HOME,
    BUTTON_START_PAUSE,
    FPS as GAMEPAD_FPS,
    HOME_MOVE_DURATION_S,
    PORT,
    auto_detect_port,
    move_to_home,
    step_targets,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

# ---- Config: edit these for your setup / task ----
CAMERA_INDEX = 1
FPS = 30
MIN_EPISODE_TIME_S = 2.0  # ignore N presses before this, to prevent accidental instant episodes
NUM_EPISODES = 30
SINGLE_TASK = "Pick up the small object from location A and drop it in location B"
REPO_ID = "local/so101_pick_place_v4_gamepad"


def init_recording_keyboard_listener():
    """Same as lerobot's init_keyboard_listener(), plus a D=delete control."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "delete_requested": False,
    }

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)
        elif key == "d":
            print("D pressed. Marking last saved episode for deletion...")
            events["delete_requested"] = True

    listener = create_key_listener(on_key, controls_help="N=next, R=re-record, D=delete last, Q=quit")
    return listener, events


def main():
    init_logging()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERROR: no gamepad detected.")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Connected gamepad: {js.get_name()}")

    port = auto_detect_port(fallback=PORT)
    robot_config = SOFollowerRobotConfig(
        port=port,
        cameras={
            "top": OpenCVCameraConfig(index_or_path=CAMERA_INDEX, fps=FPS, width=640, height=480),
        },
    )
    robot = SOFollower(robot_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    dataset = LeRobotDataset.create(
        REPO_ID,
        FPS,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_threads=2,
    )

    print(f"Connecting to arm on {port} and camera index {CAMERA_INDEX}...")
    robot.connect()
    print("Connected. Torque enabled -- the gamepad drives the arm.")

    home_position = dict(robot.bus.sync_read("Present_Position"))
    current_targets = dict(home_position)
    arm_running = False  # gated by Cross, independent of recording state

    listener, events = init_recording_keyboard_listener()

    print(f"""
=======================================================
  Gamepad recording -- task: {SINGLE_TASK!r}
  Dataset: {dataset.root}

  Cross (button 0)    - toggle arm motion on/off
  Triangle (button 3) - return arm to start position (best used
                         between episodes, not mid-episode)

  N / right - when waiting: start recording now
              when recording: end episode now (success)
  R / left  - discard and re-record the CURRENT episode
  D         - when waiting: mark the last saved episode for deletion
              (press repeatedly to go further back); actually removed
              from the dataset at the end of the session
  Q / esc   - stop recording entirely

  Episodes have no time limit -- they run until you press N.
  There is NO auto-start between episodes -- recording only begins
  once you press N, so take as long as you need to reset the scene.

  Press Cross now to enable arm motion before you start.
=======================================================
""")

    control_interval = 1 / FPS
    gamepad_interval = 1 / GAMEPAD_FPS
    episodes_to_delete: list[int] = []

    def poll_gamepad_and_drive_arm() -> None:
        """Handle Cross/Triangle and, if running, advance current_targets and write to the robot.

        Call this every iteration of any wait loop so the arm stays live
        even while an episode isn't being recorded.
        """
        nonlocal arm_running, current_targets
        pygame.event.pump()

        if js.get_button(BUTTON_START_PAUSE):
            while js.get_button(BUTTON_START_PAUSE):
                pygame.event.pump()
                time.sleep(0.01)
            arm_running = not arm_running
            if arm_running:
                current_targets = dict(robot.bus.sync_read("Present_Position"))
                print("Arm motion ON.")
            else:
                print("Arm motion OFF -- holding position.")

        if js.get_button(BUTTON_HOME):
            while js.get_button(BUTTON_HOME):
                pygame.event.pump()
                time.sleep(0.01)
            if arm_running:
                print("Triangle pressed -- returning to start position...")
                current_targets = move_to_home(
                    robot, current_targets, home_position, GAMEPAD_FPS, HOME_MOVE_DURATION_S
                )
                print("At start position.")

        if arm_running:
            step_targets(js, current_targets, gamepad_interval)
            robot.bus.sync_write("Goal_Position", current_targets)

    def delete_last_episode() -> None:
        nonlocal recorded_episodes, just_deleted
        candidates = [i for i in range(dataset.num_episodes) if i not in episodes_to_delete]
        if not candidates:
            print("No recorded episodes left to delete.")
            return
        last_idx = candidates[-1]
        episodes_to_delete.append(last_idx)
        recorded_episodes = max(0, recorded_episodes - 1)
        just_deleted = True
        print(
            f"Marked episode {last_idx} for deletion ({len(episodes_to_delete)} pending, "
            f"removed at the end of the session). Press N to re-record episode "
            f"{recorded_episodes + 1}/{NUM_EPISODES}."
        )

    def wait_until_ready(message: str) -> None:
        """Block until N is pressed (or recording is stopped entirely), while keeping the arm live."""
        events["exit_early"] = False  # drain any stale/leftover press before waiting
        print(message)
        while not events["exit_early"] and not events["stop_recording"]:
            poll_gamepad_and_drive_arm()
            if events["delete_requested"]:
                events["delete_requested"] = False
                delete_last_episode()
            time.sleep(gamepad_interval)
        events["exit_early"] = False

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            just_deleted = False
            wait_until_ready(
                "Press Cross to enable the arm, get the scene into the starting position, "
                "then press N when ready..."
            )
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
                episode_num = recorded_episodes + 1
                if just_deleted:
                    log_say(f"Re-recording episode {episode_num}", True)
                    just_deleted = False
                else:
                    log_say(f"Recording episode {episode_num}", True)
                start_t = time.perf_counter()
                timestamp = 0.0
                while True:  # no time limit -- runs until N/R ends it
                    loop_start = time.perf_counter()
                    if events["exit_early"]:
                        if timestamp >= MIN_EPISODE_TIME_S:
                            events["exit_early"] = False
                            break
                        events["exit_early"] = False
                        events["rerecord_episode"] = False
                        print(f"(Press ignored -- wait at least {MIN_EPISODE_TIME_S:.0f}s into the episode.)")

                    poll_gamepad_and_drive_arm()

                    obs = robot.get_observation()
                    obs_processed = robot_observation_processor(obs)
                    observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

                    # The gamepad's commanded goal (not the arm's actual position) is
                    # the action -- the two can differ slightly since the servo takes
                    # a moment to catch up to a new goal.
                    action_values = {f"{joint}.pos": pos for joint, pos in current_targets.items()}
                    action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)

                    frame = {**observation_frame, **action_frame, "task": SINGLE_TASK}
                    dataset.add_frame(frame)

                    dt_s = time.perf_counter() - loop_start
                    precise_sleep(max(control_interval - dt_s, 0.0))
                    timestamp = time.perf_counter() - start_t

                if events["rerecord_episode"]:
                    log_say("Re-record episode", True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    wait_until_ready("Reset the scene, then press N when ready to re-record...")
                    continue

                print("Saving episode... (don't press N yet)")
                dataset.save_episode()
                recorded_episodes += 1
                print(f"Saved episode {recorded_episodes}/{NUM_EPISODES}")

                if not events["stop_recording"] and recorded_episodes < NUM_EPISODES:
                    wait_until_ready("Reset the scene, then press N when ready for the next episode...")
    finally:
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        pygame.joystick.quit()
        pygame.quit()
        print(f"\nRecording session done. {dataset.num_episodes} episodes saved to {dataset.root}")

        if episodes_to_delete:
            print(f"\nCleaning up {len(episodes_to_delete)} deleted episode(s): {sorted(episodes_to_delete)}...")
            dataset_root = dataset.root
            finalized = LeRobotDataset(REPO_ID)
            staging_dir = dataset_root.parent / f"{dataset_root.name}_cleanup_staging"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            delete_episodes(finalized, sorted(episodes_to_delete), output_dir=staging_dir, repo_id=REPO_ID)
            shutil.rmtree(dataset_root)
            shutil.move(str(staging_dir), str(dataset_root))
            cleaned = LeRobotDataset(REPO_ID)
            print(f"Done. Cleaned dataset now has {cleaned.num_episodes} episodes at {dataset_root}")
        else:
            print(f"Done. Recorded {dataset.num_episodes} episodes to {dataset.root}")


if __name__ == "__main__":
    main()
