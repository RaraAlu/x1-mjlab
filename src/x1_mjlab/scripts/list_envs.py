"""Registers the custom BDX-R task before running mjlab's training pipeline."""

import x1_mjlab.tasks  # noqa: F401 to register environments
from mjlab.scripts.list_envs import main

if __name__ == "__main__":
    main()
