import mteb

from .tasks.BrowseCompPlus import BrowseCompPlus
from .tasks.ToolRet import ToolRet
from .tasks.LongMemEval import LongMemEval
from .tasks.LoCoMo import LoCoMo
from .tasks.AITQA import AITQA
from .tasks.CosQAPlus import CosQAPlus
from .tasks.FRAMES import FRAMES
from .tasks.FreshStack import FreshStack
from .tasks.LeetCode import LeetCode
from .tasks.MTRAG import MTRAG
from .tasks.MuSiQue import MuSiQue


LOCAL_REGISTRY = {
    c.metadata.name: c for c in [
        BrowseCompPlus,ToolRet,LoCoMo,LongMemEval,AITQA,CosQAPlus,FRAMES,FreshStack,LeetCode,MTRAG,MuSiQue
    ]
}
_builtin_tasks = set(mteb.overview.TASKS_REGISTRY.keys())
assert all(k not in _builtin_tasks for k in LOCAL_REGISTRY.keys())
mteb.overview.TASKS_REGISTRY.update(LOCAL_REGISTRY)
