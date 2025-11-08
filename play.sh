#!/bin/bash

# uv run python -m mjlab.scripts.play Mjlab-Velocity-Flat-Agibot-X1-Play \
#     --checkpoint-file ~/Work/mjlab/logs/rsl_rl/x1_velocity/2025-11-04_13-53-18/model_4150.pt \
#     --num-envs 1 \
    # --agent joystick \
    # --viewer viser
uv run python -m x1_mjlab.scripts.play_joystick Mjlab-Velocity-Flat-Agibot-X1-Play \
    --checkpoint-file ~/Work/mjlab/logs/rsl_rl/x1_velocity/2025-11-08_08-46-25/model_6500.pt \
    --num-envs 1 \
    --agent joystick \
    # --viewer viser
# uv run python -m mjlab.scripts.play_joystick Mjlab-Velocity-Flat-Unitree-G1-Play \
#     --checkpoint-file ~/Work/mjlab/logs/rsl_rl/g1_velocity/2025-11-03_16-36-06/model_2000.pt \
#     --num-envs 1 \
#     --agent joystick \


