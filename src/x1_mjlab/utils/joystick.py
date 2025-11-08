"""
手柄控制模块 - 用于通过游戏手柄控制强化学习智能体

提供混合控制模式:手柄输入高层速度指令 → RL策略生成关节动作
"""

from typing import Optional, TYPE_CHECKING
import torch
import pygame

# 类型检查时导入,运行时不导入
if TYPE_CHECKING:
  from pygame.joystick import JoystickType


class JoystickConfig:
  """手柄配置参数"""

  def __init__(
      self,
      joystick_id: int = 0,
      deadzone: float = 0.15,
      max_lin_vel: float = 1.5,
      max_ang_vel: float = 1.0,
      debug: bool = True,
      # 轴映射配置(可根据不同手柄调整)
      axis_forward: int = 1,  # 前进/后退轴
      axis_strafe: int = 0,  # 左右平移轴
      axis_rotate: int = 3,  # 转向轴
      invert_forward: bool = True,  # 是否反转前进轴
      invert_strafe: bool = True,  # 是否反转平移轴
      invert_rotate: bool = True,  # 是否反转旋转轴
  ):
    self.joystick_id = joystick_id
    self.deadzone = deadzone
    self.max_lin_vel = max_lin_vel
    self.max_ang_vel = max_ang_vel
    self.debug = debug

    self.axis_forward = axis_forward
    self.axis_strafe = axis_strafe
    self.axis_rotate = axis_rotate

    self.invert_forward = invert_forward
    self.invert_strafe = invert_strafe
    self.invert_rotate = invert_rotate


class PolicyJoystick:
  """
  使用游戏手柄控制机器人的策略(混合控制模式)

  策略架构:
      手柄 → 高层运动指令(vx, vy, vyaw) → 训练好的RL策略 → 关节动作

  支持的手柄映射(可通过 JoystickConfig 自定义):
  - 左摇杆垂直轴: 前进/后退速度 (vx)
  - 左摇杆水平轴: 左右平移速度 (vy)
  - 右摇杆水平轴: 转向速度 (vyaw)
  """

  def __init__(
      self,
      trained_policy,
      device: str,
      config: Optional[JoystickConfig] = None,
  ):
    """
    初始化手柄控制策略

    Args:
        trained_policy: 训练好的RL策略对象
        device: PyTorch设备 (cuda/cpu)
        config: 手柄配置对象,如果为None则使用默认配置
    """
    self.trained_policy = trained_policy
    self.device = device
    self.config = config or JoystickConfig()

    self.step_count = 0
    # 修改类型提示,使用具体类型而不是Optional
    self.joystick: "pygame.joystick.JoystickType"
    self.num_axes: int
    self.num_buttons: int

    # 初始化手柄
    self._initialize_joystick()

  def _initialize_joystick(self):
    """初始化 pygame 和手柄硬件"""
    pygame.init()
    pygame.joystick.init()

    # 检查手柄连接
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
      raise RuntimeError(
          "❌ No joystick detected! Please connect a controller.")

    if self.config.joystick_id >= joystick_count:
      raise ValueError(
          f"❌ Joystick ID {self.config.joystick_id} not found. "
          f"Available IDs: 0-{joystick_count-1}"
      )

    # 连接手柄 - 添加类型注释以帮助类型检查器
    self.joystick = pygame.joystick.Joystick(
        self.config.joystick_id)  # type: ignore[assignment]
    self.joystick.init()

    # 获取手柄信息
    self.num_axes = self.joystick.get_numaxes()
    self.num_buttons = self.joystick.get_numbuttons()

    # 打印初始化信息
    self._print_info()

    # 测试手柄读取
    if self.config.debug:
      self._test_joystick_read()

  def _print_info(self):
    """打印手柄信息和控制映射"""
    print(f"\n{'='*60}")
    print(f"🎮 Joystick Control Mode (Hybrid)")
    print(f"{'='*60}")
    print(f"  Controller: {self.joystick.get_name()}")
    print(f"  Axes: {self.num_axes}")
    print(f"  Buttons: {self.num_buttons}")
    print(f"")
    print(f"  📋 Control Mapping:")
    print(f"     Axis {self.config.axis_forward} → Forward/Backward (vx)")
    print(f"     Axis {self.config.axis_strafe}  → Left/Right (vy)")
    print(f"     Axis {self.config.axis_rotate}  → Rotate (vyaw)")
    print(f"")
    print(f"  ⚙️  Parameters:")
    print(f"     Max Linear Velocity:  {self.config.max_lin_vel} m/s")
    print(f"     Max Angular Velocity: {self.config.max_ang_vel} rad/s")
    print(f"     Deadzone: {self.config.deadzone}")
    print(
        f"     Debug Mode: {'✅ ENABLED' if self.config.debug else '❌ DISABLED'}"
    )
    print(f"{'='*60}\n")

  def _test_joystick_read(self):
    """测试手柄读取功能"""
    try:
      pygame.event.pump()
      print("🔍 Testing initial joystick read...")
      print("  ✅ pygame.event.pump() successful")

      # 读取所有轴的值
      print(f"  📊 All axes values:")
      for i in range(self.num_axes):
        axis_val = self.joystick.get_axis(i)
        print(f"     Axis {i}: {axis_val:+.4f}")

      # 读取所有按钮状态
      print(f"  🔘 All button states:")
      pressed_buttons = [
          i for i in range(self.num_buttons) if self.joystick.get_button(i)
      ]
      if pressed_buttons:
        print(f"     Pressed: {pressed_buttons}")
      else:
        print(f"     None pressed")

      print("  ✅ Joystick test completed\n")
    except Exception as e:
      print(f"  ❌ Joystick test failed: {e}\n")

  def _apply_deadzone(self, value: float) -> float:
    """
    应用死区,避免摇杆漂移

    Args:
        value: 原始轴值 [-1, 1]

    Returns:
        应用死区后的值,重新映射到 [-1, 1]
    """
    if abs(value) < self.config.deadzone:
      return 0.0
    # 重新映射到 [-1, 1] 范围
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - self.config.deadzone) / (1.0 - self.config.deadzone)

  def _read_axis(self, axis_id: int, invert: bool = False) -> float:
    """
    读取指定轴的值并应用配置

    Args:
        axis_id: 轴编号
        invert: 是否反转

    Returns:
        处理后的轴值
    """
    if axis_id >= self.num_axes:
      return 0.0

    value = self.joystick.get_axis(axis_id)
    if invert:
      value = -value
    return value

  def _read_velocity_command(self) -> torch.Tensor:
    """
    读取手柄状态并转换为速度指令

    Returns:
        velocity_cmd: 形状为 (3,) 的张量 [vx, vy, vyaw]
    """
    # 更新手柄状态
    pygame.event.pump()

    # 读取摇杆轴
    raw_vx = self._read_axis(self.config.axis_forward,
                             self.config.invert_forward)
    raw_vy = self._read_axis(self.config.axis_strafe,
                             self.config.invert_strafe)
    raw_vyaw = self._read_axis(
        self.config.axis_rotate, self.config.invert_rotate)

    # 调试打印:详细信息
    if self.config.debug and self.step_count % 50 == 0:
      self._print_debug_info(raw_vx, raw_vy, raw_vyaw)

    # 应用死区
    vx_normalized = self._apply_deadzone(raw_vx)
    vy_normalized = self._apply_deadzone(raw_vy)
    vyaw_normalized = self._apply_deadzone(raw_vyaw)

    # 缩放到实际速度
    vx = vx_normalized * self.config.max_lin_vel
    vy = vy_normalized * self.config.max_lin_vel
    vyaw = vyaw_normalized * self.config.max_ang_vel

    # 实时简化打印
    if self.config.debug:
      self._print_live_status(vx, vy, vyaw)

    return torch.tensor([vx, vy, vyaw], device=self.device)

  def _print_debug_info(self, raw_vx: float, raw_vy: float, raw_vyaw: float):
    """打印详细调试信息(每50步)"""
    print(f"\n{'─'*60}")
    print(f"🔍 DEBUG [Step {self.step_count}] - Joystick State")
    print(f"{'─'*60}")
    print(f"  📥 Raw Axis Values:")
    for i in range(min(6, self.num_axes)):
      print(f"     Axis {i}: {self.joystick.get_axis(i):+.4f}")

    print(f"\n  🎯 Mapped Raw Values (before deadzone):")
    print(f"     raw_vx:   {raw_vx:+.4f}")
    print(f"     raw_vy:   {raw_vy:+.4f}")
    print(f"     raw_vyaw: {raw_vyaw:+.4f}")
    print(f"{'─'*60}")

  def _print_live_status(self, vx: float, vy: float, vyaw: float):
    """打印实时状态(简化版)"""
    if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
      print(
          f"\r🎮 [Step {self.step_count:4d}] Command: "
          f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}  ",
          end="",
      )
    elif self.step_count % 100 == 0:
      print(
          f"\r🎮 [Step {self.step_count:4d}] Command: "
          f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} (idle)",
          end="",
      )

  def __call__(self, obs: dict) -> torch.Tensor:
    """
    策略调用接口(混合控制模式)

    流程:
      1. 从手柄读取速度指令
      2. 修改观测中的 command 字段
      3. 调用训练好的策略生成关节动作

    Args:
        obs: 环境观测值字典,包含 'policy' 键

    Returns:
        动作张量,由训练好的策略生成
    """
    self.step_count += 1

    # 读取手柄指令
    velocity_cmd = self._read_velocity_command()

    # 修改观测中的 command 部分
    obs_policy = obs["policy"].clone()
    num_envs = obs_policy.shape[0]

    # 将手柄指令广播到所有环境
    velocity_cmd_batch = velocity_cmd.unsqueeze(0).repeat(num_envs, 1)

    # 替换观测中的 command 部分(假设最后3个维度)
    obs_policy[:, -3:] = velocity_cmd_batch

    # 构建修改后的观测
    modified_obs = {"policy": obs_policy}
    if "critic" in obs:
      modified_obs["critic"] = obs["critic"]

    # 调用训练好的策略
    action = self.trained_policy(modified_obs)

    # 详细调试打印
    if self.config.debug and self.step_count % 50 == 0:
      self._print_policy_debug(obs, modified_obs, action)

    return action

  def _print_policy_debug(self, obs: dict, modified_obs: dict, action: torch.Tensor):
    """打印策略调用的详细调试信息"""
    print(f"\n{'═'*60}")
    print(f"🧠 DEBUG [Step {self.step_count}] - Policy Call")
    print(f"{'═'*60}")
    print(f"  📊 Observation Info:")
    print(f"     obs['policy'] shape: {obs['policy'].shape}")
    print(
        f"     Modified command: {modified_obs['policy'][0, -3:].cpu().numpy()}")
    print(f"\n  🎯 Policy Output:")
    print(f"     action shape: {action.shape}")
    print(
        f"     action range: [{action.min().item():.4f}, {action.max().item():.4f}]"
    )
    print(f"     action mean: {action.mean().item():.4f}")
    print(f"{'═'*60}\n")

  def reset(self):
    """重置策略状态"""
    self.step_count = 0

  def __del__(self):
    """析构函数:清理 pygame 资源"""
    # 添加检查避免在未初始化时出错
    if hasattr(self, "joystick"):
      self.joystick.quit()
    pygame.quit()
    if hasattr(self, "config") and self.config.debug:
      print("\n🎮 Joystick disconnected and cleaned up")


def print_joystick_test_prompt():
  """打印手柄测试提示信息"""
  print("\n" + "=" * 60)
  print("🎮 JOYSTICK TEST MODE")
  print("=" * 60)
  print("  Please move the joystick sticks to test:")
  print("  - Left stick: Should control vx (forward/backward) and vy (left/right)")
  print("  - Right stick: Should control vyaw (rotation)")
  print("  ")
  print("  The debug output will show:")
  print("  1. Raw axis values from joystick")
  print("  2. Values after deadzone filtering")
  print("  3. Final velocity commands")
  print("  4. How observations are modified")
  print("  5. Actions generated by the policy")
  print("=" * 60)
  input("\n  Press ENTER to start... ")
  print("\n")
