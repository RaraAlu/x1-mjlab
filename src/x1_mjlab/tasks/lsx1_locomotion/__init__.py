import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-DarkSea-X1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DarkSeaX1RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:DarkSeaX1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-DarkSea-X1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DarkSeaX1RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:DarkSeaX1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-DarkSea-X1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DarkSeaX1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:DarkSeaX1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-DarkSea-X1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DarkSeaX1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:DarkSeaX1PPORunnerCfg",
  },
)
