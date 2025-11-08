
# LSX1 in MjLab

This repository showcase the implementation of the LSX1 robot into [MjLab](https://github.com/mujocolab/mjlab). 

> [!WARNING]
> As MjLab is still in early development, this repository may be impacted by breaking changes. If an issue were to arise when running one of the scripts, feel free to open an issue or contribute to the project. Thanks you for your understanding!

## 🤖 What's MjLab?

MjLab is a project to have the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) API using [MjWarp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html) as the backend. If you’re wondering about the motivation behind it or how it differs from Newton, you can learn more about it [here](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

## 🚀 Quickstart

Clone the repository.

```bash
git clone https://github.com/RaraAlu/x1-mjlab.git && cd X1-MjLab
```

List available environments.

```bash
uv run x1_list_envs
```

### Velocity Tracking

Train the policy.

```bash
./train.sh
```
经过试验新提交合并的奖励函数中，线速度 xy跟踪奖励 不如 最初版本对于 x1 来的有效，如果克隆本仓库，记得查看历史提交并将最新的线速度xy奖励函数回退到最初版本


Evaluate the policy.

```bash
./play.sh
```
