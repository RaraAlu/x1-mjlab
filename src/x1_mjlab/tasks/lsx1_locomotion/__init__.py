import gymnasium as gym

from .env_cfgs import DARKSEA_X1_FLAT_ENV_CFG, DARKSEA_X1_ROUGH_ENV_CFG

gym.register(
  id="Mjlab-Velocity-Rough-DarkSea-X1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": DARKSEA_X1_ROUGH_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:DarkSeaX1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-DarkSea-X1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": DARKSEA_X1_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:DarkSeaX1PPORunnerCfg",
  },
)
