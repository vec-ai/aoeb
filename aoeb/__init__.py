import mteb

from .tasks.BrowseCompPlus import BrowseCompPlus
from .tasks.ToolRet import ToolRet
from .tasks.LongMemEval import LongMemEval
from .tasks.LoCoMo import LoCoMo
# from .tasks.AITQA import AITQA
# from .tasks.CosQAPlus import CosQAPlus
from .tasks.FRAMES import FRAMES
from .tasks.FreshStack import FreshStack
# from .tasks.LeetCode import LeetCode
# from .tasks.MTRAG import MTRAG
from .tasks.MuSiQue import MuSiQue
from .tasks.MyLLaVA import MyLLaVA
from .tasks.DocVQA import DocVQA
from .tasks.ArxivQA import ArxivQA
from .tasks.BRIGHT import BRIGHT
from .tasks.SweBenchMultimodal import SweBenchMultimodal
from .tasks.Design2Code import Design2Code
from .tasks.MRMRTheorem import MRMRTheorem
from .tasks.MRMRTraffic import MRMRTraffic
from .tasks.ChartMimic import ChartMimic
from .tasks.ReFocus import ReFocus
from .tasks.MVRBComposedScreenshotRetrieval import MVRBComposedScreenshotRetrievalKnowledgeRelation,\
MVRBComposedScreenshotRetrievalNewsToWiki,MVRBComposedScreenshotRetrievalProductDiscovery,\
MVRBComposedScreenshotRetrievalWikiToProduct



from mteb.get_tasks import _TASKS_REGISTRY



LOCAL_REGISTRY = {
    c.metadata.name: c for c in [
        BrowseCompPlus,ToolRet,
        # AITQA,CosQAPlus,
        FRAMES,FreshStack,
        # LeetCode,MTRAG,
        MuSiQue,
        # MyLLaVA,DocVQA,
        ArxivQA,
        MyLLaVA,DocVQA,SweBenchMultimodal,
        LoCoMo,LongMemEval,
        BRIGHT,
        Design2Code,
        ChartMimic,
        MRMRTheorem,
        MRMRTraffic,
        ReFocus,
        MVRBComposedScreenshotRetrievalKnowledgeRelation,
        MVRBComposedScreenshotRetrievalNewsToWiki,
        MVRBComposedScreenshotRetrievalProductDiscovery,
        MVRBComposedScreenshotRetrievalWikiToProduct,
    ]
}
# _builtin_tasks = set(mteb.overview.TASKS_REGISTRY.keys())
# assert all(k not in _builtin_tasks for k in LOCAL_REGISTRY.keys())
# mteb.overview.TASKS_REGISTRY.update(LOCAL_REGISTRY)

# for mteb 2.1.0
# 获取所有内置任务名称
_builtin_tasks = set(_TASKS_REGISTRY.keys())

# 检查本地任务是否与内置任务冲突
assert all(k not in _builtin_tasks for k in LOCAL_REGISTRY.keys()), \
    f"Task name conflict detected: {set(LOCAL_REGISTRY.keys()) & _builtin_tasks}"

# 注册自定义任务
_TASKS_REGISTRY.update(LOCAL_REGISTRY)
