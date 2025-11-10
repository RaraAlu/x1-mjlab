"""Agibot X1 机器人常数配置模块。

本模块定义了Agibot X1双足机器人的所有核心配置参数，包括：
- 机器人模型文件和资源管理
- 执行器（电机）配置和参数
- 机器人姿态关键帧定义
- 碰撞模型配置
"""

from pathlib import Path
import math

import mujoco

from x1_mjlab import MJLAB_X1_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
    ElectricActuator,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg


##
# MJCF 和资源配置
##

# X1机器人的MJCF（MuJoCo XML格式）模型文件路径
# MJCF是MuJoCo物理引擎的标准模型描述格式
X1_XML: Path = (
    MJLAB_X1_SRC_PATH
    / "robot_asset"
    / "robots"
    / "ls_x1"
    / "xmls"
    / "x1.xml"
)
# 检查模型文件是否存在
assert X1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """
  获取机器人的所有资源文件（网格模型等）。

  参数:
    meshdir: 网格文件目录路径

  返回:
    字典，键为资源名称，值为资源的字节数据
  """
  assets: dict[str, bytes] = {}
  # 从X1模型文件的资源目录中更新资源字典
  update_assets(assets, X1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:  # type: ignore
  """
  获取MuJoCo规范对象，用于物理模拟。

  返回:
    MjSpec对象，包含完整的机器人模型规范和资源
  """
  # 从MJCF文件加载模型规范
  spec = mujoco.MjSpec.from_file(str(X1_XML))  # type: ignore
  # 关联机器人的资源文件（网格、纹理等）
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# 执行器配置（基于智元PowerFlow电机）
##
ARMATURE_R86_3 = 0.022
ARMATURE_R86_2 = 0.015
ARMATURE_R52 = 0.0085


ACTUATOR_R86_3 = ElectricActuator(
    reflected_inertia=ARMATURE_R86_3,  # 反射惯量
    velocity_limit=150,  # 速度限制
    effort_limit=8,  # 扭矩限制
)
ACTUATOR_R86_2 = ElectricActuator(
    reflected_inertia=ARMATURE_R86_2,  # 反射惯量
    velocity_limit=50,  # 速度限制
    effort_limit=24,  # 扭矩限制
)
ACTUATOR_R52 = ElectricActuator(
    reflected_inertia=ARMATURE_R52,  # 反射惯量
    velocity_limit=80,  # 速度限制
    effort_limit=10,  # 扭矩限制
)


NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_R86_3 = ARMATURE_R86_3 * NATURAL_FREQ**2
STIFFNESS_R86_2 = ARMATURE_R86_2 * NATURAL_FREQ**2
STIFFNESS_R52 = ARMATURE_R52 * NATURAL_FREQ**2

DAMPING_R86_3 = 2.0 * DAMPING_RATIO * ARMATURE_R86_3 * NATURAL_FREQ
DAMPING_R86_2 = 2.0 * DAMPING_RATIO * ARMATURE_R86_2 * NATURAL_FREQ
DAMPING_R52 = 2.0 * DAMPING_RATIO * ARMATURE_R52 * NATURAL_FREQ



# R86-3 执行器配置对象
X1_ACTUATOR_R86_3 = ActuatorCfg(
    # 关节名称匹配表达式（正则表达式）
    joint_names_expr=(
        ".*_hip_pitch_joint",  # 髋关节俯仰
        ".*_hip_roll_joint",   # 髋关节翻滚
        ".*_knee_pitch_joint",  # 膝关节俯仰
    ),
    effort_limit=ACTUATOR_R86_3.effort_limit,  # 扭矩限制
    armature=ACTUATOR_R86_3.reflected_inertia,  # 电枢惯量
    stiffness=STIFFNESS_R86_3,  # 刚度
    damping=DAMPING_R86_3,  # 阻尼
)



# R86-2 执行器配置对象
X1_ACTUATOR_R86_2 = ActuatorCfg(
    joint_names_expr=(
        # 手臂关节
        ".*_shoulder_pitch_joint",  # 肩关节俯仰
        ".*_shoulder_roll_joint",   # 肩关节翻滚
        # 腿部关节
        ".*_hip_yaw_joint",  # 髋关节偏航
        # 躯干关节（3个）
        "waist_yaw_joint",    # 腰部偏航
        "waist_roll_joint",   # 腰部翻滚
        "waist_pitch_joint",  # 腰部俯仰
    ),
    effort_limit=ACTUATOR_R86_2.effort_limit,
    armature=ACTUATOR_R86_2.reflected_inertia,
    stiffness=STIFFNESS_R86_2,
    damping=DAMPING_R86_2,
)


# R52 执行器配置对象
X1_ACTUATOR_R52 = ActuatorCfg(
    joint_names_expr=(
        # 手臂关节
        ".*_shoulder_yaw_joint",   # 肩关节偏航
        ".*_elbow_pitch_joint",    # 肘关节俯仰
        ".*_elbow_yaw_joint",      # 肘关节偏航
        # 腿部关节
        ".*_ankle_pitch_joint",    # 踝关节俯仰
        ".*_ankle_roll_joint",     # 踝关节翻滚
    ),
    effort_limit=ACTUATOR_R52.effort_limit,
    armature=ACTUATOR_R52.reflected_inertia,
    stiffness=STIFFNESS_R52,
    damping=DAMPING_R52,
)

X1_ACTUATOR_WAIST = ActuatorCfg(
    joint_names_expr=(
        "waist_yaw_joint",    # 腰部偏航
        "waist_roll_joint",   # 腰部翻滚
        "waist_pitch_joint",  # 腰部俯仰
    ),
    effort_limit=ACTUATOR_R86_2.effort_limit,
    armature=ACTUATOR_R86_2.reflected_inertia,
    stiffness=STIFFNESS_R86_2,
    damping=DAMPING_R86_2,
)

X1_ACTUATOR_ANKLE = ActuatorCfg(
    joint_names_expr=(
        ".*_ankle_pitch_joint",    # 踝关节俯仰
        ".*_ankle_roll_joint",     # 踝关节翻滚
    ),
    effort_limit=ACTUATOR_R52.effort_limit,
    armature=ACTUATOR_R52.reflected_inertia,
    stiffness=STIFFNESS_R52,
    damping=DAMPING_R52,
)

##
# 关键帧配置（机器人预定义姿态）
##

# 【HOME姿态】：直立站立状态
# 特点：腿部伸直、膝关节略微弯曲、手臂放松
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    # 位置：设定机器人质心的初始位置（单位：米）
    # X1机器人身高130cm，质心约在0.65m处
    pos=(0, 0, 0.65),

    # 关节位置（角度单位：弧度）
    joint_pos={
        # --- 躯干关节 ---
        "waist_yaw_joint": 0.0,    # 腰部偏航：0度（面向前方）
        "waist_roll_joint": 0.0,   # 腰部翻滚：0度（不倾斜）
        "waist_pitch_joint": 0.0,  # 腰部俯仰：0度（竖直）

        # --- 腿部关节 ---
        ".*_hip_pitch_joint": 0.0,     # 髋关节俯仰：0度（向前）
        ".*_hip_roll_joint": 0.0,      # 髋关节翻滚：0度（不打开腿）
        ".*_hip_yaw_joint": 0.0,       # 髋关节偏航：0度
        ".*_knee_pitch_joint": 0.0,    # 膝关节俯仰：0度（伸直）
        ".*_ankle_pitch_joint": 0.0,   # 踝关节俯仰：0度
        ".*_ankle_roll_joint": 0.0,    # 踝关节翻滚：0度

        # --- 手臂关节（收回状态）---
        ".*_shoulder_pitch_joint": 0.16,   # 肩关节俯仰
        ".*_shoulder_roll_joint": -0.1,    # 肩关节翻滚
        ".*_shoulder_yaw_joint": 0.0,      # 肩关节偏航
        ".*_elbow_pitch_joint": 0.3,       # 肘关节俯仰（弯曲）
        ".*_elbow_yaw_joint": 0.0,         # 肘关节偏航
    },
    # 所有关节的初始速度均为0（静止状态）
    joint_vel={".*": 0.0},
)

# 【READY姿态】：准备就绪/略微下蹲状态
# 特点：质心降低、姿态更稳定，便于快速运动
READY_KEYFRAME = EntityCfg.InitialStateCfg(
    # 位置：质心略微降低（下蹲状态）
    pos=(0, 0, 0.63),

    # 关节位置配置（与HOME姿态相同）
    joint_pos={
        # --- 躯干关节 ---
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,

        # --- 腿部关节 ---
        ".*_hip_pitch_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        ".*_knee_pitch_joint": 0.0,
        ".*_ankle_pitch_joint": 0.0,
        ".*_ankle_roll_joint": 0.0,

        # --- 手臂关节 ---
        ".*_shoulder_pitch_joint": 0.16,
        ".*_shoulder_roll_joint": -0.1,
        ".*_shoulder_yaw_joint": 0.0,
        ".*_elbow_pitch_joint": 0.3,
        ".*_elbow_yaw_joint": 0.0,
    },
    # 所有关节的初始速度均为0
    joint_vel={".*": 0.0},
)

##
# 碰撞配置
##

# 脚部几何体的正则表达式匹配模式
# 用于识别机器人模型中的所有脚部碰撞网格
FOOT_GEOM_NAMES_REGEX = r"^(left|right)_foot[1-7]_collision$"
# 脚部摩擦系数（MuJoCo标准值）
FOOT_FRICTION = 0.6

# 【FULL_COLLISION】：完整碰撞配置（包括自碰撞检测）
# 自碰撞：检测机器人自身不同部分之间的碰撞
# condim参数说明：
#   - condim=1：1维接触（仅法向力，用于检测自碰撞）
#   - condim=3：3维接触（法向力+切向摩擦，用于脚部与地面接触）
FULL_COLLISION = CollisionCfg(
    # 所有包含"_collision"的几何体参与碰撞检测
    geom_names_expr=(".*_collision",),
    # 脚部使用3维接触，其他部分使用1维接触
    condim={FOOT_GEOM_NAMES_REGEX: 3, ".*_collision": 1},
    # 脚部碰撞优先级最高
    priority={FOOT_GEOM_NAMES_REGEX: 1},
    # 脚部使用自定义摩擦系数
    friction={FOOT_GEOM_NAMES_REGEX: (FOOT_FRICTION,)},
)

# 【FULL_COLLISION_WITHOUT_SELF】：完整碰撞配置（不包括自碰撞检测）
# contype=0, conaffinity=1：禁用自碰撞
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,      # 接触类型：0表示不与自身碰撞
    conaffinity=1,  # 亲和力：只与环境碰撞
    condim={FOOT_GEOM_NAMES_REGEX: 3, ".*_collision": 1},
    priority={FOOT_GEOM_NAMES_REGEX: 1},
    friction={FOOT_GEOM_NAMES_REGEX: (FOOT_FRICTION,)},
)

# 【FEET_ONLY_COLLISION】：仅脚部碰撞配置
# 仅检测脚部与环境的接触，用于优化计算效率
FEET_ONLY_COLLISION = CollisionCfg(
    # 仅脚部几何体参与碰撞检测
    geom_names_expr=(FOOT_GEOM_NAMES_REGEX,),
    contype=0,      # 禁用自碰撞
    conaffinity=1,  # 仅与环境碰撞
    condim=3,       # 3维接触（包括摩擦）
    priority=1,     # 优先级
    friction=(FOOT_FRICTION,),  # 使用标准脚部摩擦系数
)

##
# 最终配置组件
##

# 机器人关节和执行器的总体配置
X1_ARTICULATION = EntityArticulationInfoCfg(
    # 使用三种不同规格的执行器
    actuators=(
        X1_ACTUATOR_R86_3,  # 高扭矩腿部电机
        X1_ACTUATOR_R86_2,  # 中扭矩腿部/躯干电机
        X1_ACTUATOR_R52,    # 低扭矩手臂电机
    ),
    # 软关节位置限制因子：允许关节达到最大范围的90%
    # 防止机器人关节碰到硬限制
    soft_joint_pos_limit_factor=0.9,
)


def get_x1_robot_cfg() -> EntityCfg:
  """Get a fresh X1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=READY_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X1_ARTICULATION,
  )

# X1机器人配置对象，用于外部导入
LSX1_ROBOT_CFG = get_x1_robot_cfg()


# 【动作规模计算】：强化学习标准化动作空间的必要参数
# 为每个关节计算动作缩放因子，使得归一化动作（-1到1）映射到合适的扭矩范围
LSX1_ACTION_SCALE: dict[str, float] = {}
# 遍历所有执行器
for a in X1_ARTICULATION.actuators:
  e = a.effort_limit  # 获取扭矩限制
  s = a.stiffness      # 获取刚度
  names = a.joint_names_expr  # 获取应用的关节名称列表

  # 如果扭矩限制不是字典，转换为字典格式
  if not isinstance(e, dict):
    e = {n: e for n in names}
  # 如果刚度不是字典，转换为字典格式
  if not isinstance(s, dict):
    s = {n: s for n in names}

  # 为每个关节计算动作缩放因子
  for n in names:
    if n in e and n in s and s[n]:
      # 公式：动作规模 = 0.25 × 扭矩限制 / 刚度
      # 0.25是经验系数，用于保守的扭矩分配
      LSX1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

# 【主程序】：用于快速验证机器人配置
if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  # 创建X1机器人实体
  robot = Entity(get_x1_robot_cfg())

  # 启动MuJoCo可视化查看器，检查机器人配置是否正确
  viewer.launch(robot.spec.compile())
