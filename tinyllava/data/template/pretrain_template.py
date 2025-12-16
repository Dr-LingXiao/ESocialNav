# pretrain_template.py 头部 import 区
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from ...utils.constants import *
from . import register_template   # ← 补这行（和 phi_template 一致）

from transformers import PreTrainedTokenizer
import torch


@register_template('pretrain')
@dataclass
class PretrainTemplate(Template):
    format_image_token: "Formatter" = EmptyFormatter(slot="")
    # 仍保持用户侧只有图像占位符；多轮时会重复多次 <image>
    format_user: "Formatter" = StringFormatter(slot="{{content}}")
   
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}\n")
    system: "Formatter" = EmptyFormatter(slot="")
    separator: "Formatter" = EmptyFormatter(slot=['', ''])

    def make_labels(self, input_ids, prompt, tokenizer):
        if isinstance(input_ids, torch.Tensor):
            labels = input_ids.clone()
            ids = input_ids.tolist()
        else:
            labels = copy.deepcopy(input_ids)
            ids = list(input_ids)

        img_tok = self.tokenizer_image_token("<image>", tokenizer)  # 可能是长度>1的子序列
        m = len(img_tok)
        if m == 0:
            return labels

        i = 0
        while i <= len(ids) - m:
            if ids[i:i+m] == img_tok:
                if isinstance(labels, torch.Tensor):
                    labels[i:i+m] = IGNORE_INDEX
                else:
                    labels[i:i+m] = [IGNORE_INDEX] * m
                i += m
            else:
                i += 1
        return labels

    