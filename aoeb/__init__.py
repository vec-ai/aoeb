import mteb

from .tasks.BrowseCompPlus import BrowseCompPlus
from .tasks.ToolRet import ToolRet
from .tasks.LongMemEval import LongMemEval
from .tasks.LoCoMo import LoCoMo


LOCAL_REGISTRY = {
    c.metadata.name: c for c in [
        BrowseCompPlus, ToolRet, LongMemEval, LoCoMo,
    ]
}
_builtin_tasks = set(mteb.overview.TASKS_REGISTRY.keys())
assert all(k not in _builtin_tasks for k in LOCAL_REGISTRY.keys())
mteb.overview.TASKS_REGISTRY.update(LOCAL_REGISTRY)
