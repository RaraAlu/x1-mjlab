from pathlib import Path

MJLAB_X1_SRC_PATH: Path = Path(__file__).parent

# Import tasks to register environments
from . import tasks  # noqa: F401