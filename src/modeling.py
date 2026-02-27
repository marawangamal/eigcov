# Compatibility shim: checkpoints pickled by the old repo reference src.modeling.
# Kept alive after refactor moved classes to src.vision.modeling.
from src.vision.modeling import *  # noqa: F401, F403
