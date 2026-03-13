"""用RSL-RL强化学习框架来运行和演示RL智能体的脚本。"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import mjlab.tasks  # 触发任务注册
import x1_mjlab.tasks  # 触发 x1 任务注册
from x1_mjlab.utils.joystick import (
  PolicyJoystick,
  JoystickConfig,
  print_joystick_test_prompt,
)


@dataclass(frozen=True)
class PlayConfig:
  """播放/演示配置数据类"""

  agent: Literal["zero", "random", "trained", "joystick"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["native", "viser"] = "native"

  # 🎮 手柄配置参数
  joystick_id: int = 0
  joystick_deadzone: float = 0.15
  joystick_max_lin_vel: float = 1.5
  joystick_max_ang_vel: float = 1.0
  debug_joystick: bool = True


def run_play(task: str, cfg: PlayConfig):
  """主要函数：初始化环境，加载智能体策略，并运行演示循环"""

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"[INFO]: Using device: {device}")

  # 加载环境和智能体配置
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)

  # 模式判断
  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE
  JOYSTICK_MODE = cfg.agent == "joystick"

  log_dir: Optional[Path] = None
  resume_path: Optional[Path] = None

  # 处理检查点加载
  if TRAINED_MODE or JOYSTICK_MODE:
    log_root_path = (Path("logs") / "rsl_rl" /
                     agent_cfg.experiment_name).resolve()
    print(f"[INFO]: Loading experiment from: {log_root_path}")

    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {resume_path}")
    else:
      raise ValueError(
          "`checkpoint_file` is required when using trained agent.")

    print(f"[INFO]: Loading checkpoint: {resume_path}")
    log_dir = resume_path.parent

  # 覆盖配置
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  # 确定渲染模式
  render_mode = (
      "rgb_array" if ((TRAINED_MODE or JOYSTICK_MODE)
                      and cfg.video) else None
  )
  if cfg.video and DUMMY_MODE:
    print("[WARN] Video recording with dummy agents is disabled.")

  # 创建环境
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # 创建策略
  if DUMMY_MODE:
    action_shape: tuple[int, ...] | None = env.unwrapped.action_space.shape
    if action_shape is None:
      raise RuntimeError("Action space shape is None")

    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()

    elif cfg.agent == "random":

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()

    else:
      raise ValueError(f"Unknown agent type: {cfg.agent}")

  elif JOYSTICK_MODE:
    # ✅ 使用模块化的手柄控制
    print("\n[INFO] Initializing joystick control mode...")

    # 🔧 类型安全检查：确保 log_dir 和 resume_path 不为 None
    if log_dir is None or resume_path is None:
      raise RuntimeError(
          "Checkpoint and log directory must be set for joystick mode"
      )

    # 加载训练好的策略
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    trained_policy = runner.get_inference_policy(device=device)

    # 创建手柄配置
    joystick_config = JoystickConfig(
        joystick_id=cfg.joystick_id,
        deadzone=cfg.joystick_deadzone,
        max_lin_vel=cfg.joystick_max_lin_vel,
        max_ang_vel=cfg.joystick_max_ang_vel,
        debug=cfg.debug_joystick,
    )

    # 创建手柄策略
    print("[INFO] Creating joystick policy wrapper...")
    policy = PolicyJoystick(
        trained_policy=trained_policy,
        device=env.unwrapped.device,
        config=joystick_config,
    )
    print("[INFO] ✅ Joystick policy wrapper created successfully\n")

  else:
    # 训练模式
    print("\n[INFO] Loading trained policy...")

    # 🔧 类型安全检查：确保 log_dir 和 resume_path 不为 None
    if log_dir is None or resume_path is None:
      raise RuntimeError(
          "Checkpoint and log directory must be set for trained mode"
      )

    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # 🆕 手柄测试提示
  if JOYSTICK_MODE:
    print_joystick_test_prompt()

  # 运行查看器
  if cfg.viewer == "native":
    print("[INFO] Starting Native MuJoCo Viewer...")
    NativeMujocoViewer(env, policy).run()
  elif cfg.viewer == "viser":
    print("[INFO] Starting Viser Web Viewer...")
    ViserPlayViewer(env, policy).run()
  else:
    assert_never(cfg.viewer)

  env.close()


def main():
  """主入口点"""
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
      tyro.extras.literal_type_from_choices(all_tasks),
      add_help=False,
      return_unknown_args=True,
  )

  args = tyro.cli(
      PlayConfig,
      args=remaining_args,
      default=PlayConfig(),
      prog=sys.argv[0] + f" {chosen_task}",
      config=(
          tyro.conf.AvoidSubcommands,
          tyro.conf.FlagConversionOff,
      ),
  )
  del remaining_args

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
