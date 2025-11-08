from dataclasses import dataclass

from x1_mjlab.tasks.lsx1_locomotion.rough_env_cfg import (
  DarkSeaX1RoughEnvCfg,
)


@dataclass
class DarkSeaX1FlatEnvCfg(DarkSeaX1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None


@dataclass
class DarkSeaX1FlatEnvCfg_PLAY(DarkSeaX1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    self.commands.twist.ranges.lin_vel_x = (-1.5, 2.0)
    self.commands.twist.ranges.ang_vel_z = (-0.7, 0.7)
