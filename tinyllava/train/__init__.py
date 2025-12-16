from .train import *
from .tinyllava_trainer import *
from .tinyllava_trainer_grpo import *
from .train_grpo import *
from transformers.utils import (
    is_datasets_available,
    is_flash_attn_2_available,
    is_peft_available,
)
# 兼容旧版 transformers：没有 is_rich_available 就提供一个假实现
try:
    from transformers.utils import is_rich_available
except ImportError:
    def is_rich_available() -> bool:
        return False