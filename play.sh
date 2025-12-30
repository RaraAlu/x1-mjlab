#!/bin/bash

uv run python -m x1_mjlab.scripts.play_joystick Mjlab-Velocity-Flat-DarkSea-X1 \
  --checkpoint-file logs/rsl_rl/x1_velocity/2025-12-30_22-19-21/model_1600.pt \
  --num-envs 1 \
  --agent joystick
  # --viewer viser
