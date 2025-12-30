from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import darksea_x1_flat_env_cfg, darksea_x1_rough_env_cfg
from .rl_cfg import DarkSeaX1PPORunnerCfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-DarkSea-X1",
  env_cfg=darksea_x1_rough_env_cfg(),
  play_env_cfg=darksea_x1_rough_env_cfg(play=True),
  rl_cfg=DarkSeaX1PPORunnerCfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-DarkSea-X1",
  env_cfg=darksea_x1_flat_env_cfg(),
  play_env_cfg=darksea_x1_flat_env_cfg(play=True),
  rl_cfg=DarkSeaX1PPORunnerCfg(),
  runner_cls=VelocityOnPolicyRunner,
)
