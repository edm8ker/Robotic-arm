"""Diagnostic tool: print live axis/button/hat values from the connected
gamepad so we can identify exact indices and polarity before wiring
anything to the robot.

Move ONE stick/trigger/button at a time and watch which line changes.

Usage:
  python gamepad_debug.py
"""

import time

import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No gamepad detected.")
    raise SystemExit(1)

js = pygame.joystick.Joystick(0)
js.init()
print(f"Connected: {js.get_name()}")
print(f"Axes: {js.get_numaxes()}  Buttons: {js.get_numbuttons()}  Hats: {js.get_numhats()}")
print("\nMove one stick/trigger/button at a time. Ctrl+C to quit.\n")

try:
    while True:
        pygame.event.pump()

        # Indices are printed explicitly (not just a plain list) so you read off
        # pygame's real 0-indexed axis numbers directly -- a plain list here once
        # caused every axis constant downstream to be miscounted 1-indexed.
        axes = {i: round(js.get_axis(i), 2) for i in range(js.get_numaxes())}
        buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
        hats = [js.get_hat(i) for i in range(js.get_numhats())]

        pressed_buttons = [i for i, b in enumerate(buttons) if b]

        print(f"axes={axes}  pressed_buttons={pressed_buttons}  hats={hats}", end="\r")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    pygame.joystick.quit()
    pygame.quit()
