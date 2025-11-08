from dataclasses import dataclass, replace

from x1_mjlab.robot_asset.robots.ls_x1.x1_constants import (
    LSX1_ACTION_SCALE,
    LSX1_ROBOT_CFG,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)


@dataclass
class DarkSeaX1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(LSX1_ROBOT_CFG)}

    # Constants.
    site_names = ["left_foot", "right_foot"]
    geom_names = []
    for i in range(1, 8):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 8):
      geom_names.append(f"right_foot{i}_collision")
    target_foot_height = 0.15

    # Sensors.
    feet_grund_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(
            mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    self.scene.sensors = (feet_grund_cfg, self_collision_cfg)

    # Actions.
    self.actions.joint_pos.scale = LSX1_ACTION_SCALE

    # Event.
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    # Rewards.
    self.rewards.upright.params["asset_cfg"].body_names = ["torso_link"]

    self.rewards.pose.params["std_standing"] = {".*": 0.05}

    self.rewards.pose.params["std_walking"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.15,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      # Arms.
      r".*shoulder_pitch.*": 0.35,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.25,
    }

    self.rewards.pose.params["std_running"] = {
      # Lower body.
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      # Waist.
      r".*waist_yaw.*": 0.3,
      r".*waist_roll.*": 0.1,
      r".*waist_pitch.*": 0.2,
      # Arms.
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.35,
    }

    self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
    self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["target_height"] = target_foot_height
    self.rewards.foot_clearance.params["target_height"] = target_foot_height
    self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["torso_link"]

    # curriculum
    weight_stages = [
        {"step": 0, "weight": -1e-5},
        {"step": 10000 * 24, "weight": -1e-3},
        {"step": 20000 * 24, "weight": -0.02},
      ]
    
    twist_stages = [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.8, 0.8)},
        # {"step": 10000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        # {"step": 20000 * 24, "lin_vel_x": (-2.0, 3.0)},
    ]

    self.curriculum.soft_landing_weight.params["weight_stages"] = weight_stages
    self.curriculum.command_vel.params["velocity_stages"] = twist_stages

    # Observations.
    self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

    # Terminations.
    self.terminations.illegal_contact = None

    self.viewer.body_name = "torso_link"
    self.commands.twist.viz.z_offset = 1.15
    
    self.sim.mujoco.timestep = 0.001  # 1000hz
    self.decimation = 10 # 100hz
    

@dataclass
class DarkSeaX1RoughEnvCfg_PLAY(DarkSeaX1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
