
import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *
from collections import Counter

import transformers
import torch
from torch.utils.data import Dataset


def _split_multiturn_into_samples(sample, text_preprocess, tokenizer, max_len):
    """
    将一条多轮对话展开为多条训练样本：
    每条样本仅监督其中“某一轮 assistant 的完整回复”，
    并尽可能保留该轮之前的上下文；若过长，从左侧丢弃历史 turn，
    但绝不截断这轮 assistant 的监督目标（即不做 token 级截断）。
    返回：若干个“结构与原样本一致但 conversations 被裁剪”的新样本。
    """
    convs = sample.get("conversations", [])
    if not isinstance(convs, list) or len(convs) == 0:
        return []

    def _role(c):
        # 兼容多种键
        r = c.get("role") or c.get("from")
        if r is None and "assistant" in c:
            r = "assistant"
        if r is None and "human" in c:
            r = "human"
        return r

    # 找到所有 assistant 回复的位置
    assistant_idx = [i for i, c in enumerate(convs) if _role(c) in ("assistant", "gpt", "assistant_role")]
    if not assistant_idx:
        return []  # 没有监督目标则跳过

    out = []
    base_id = str(sample.get("id", "")) if "id" in sample else None

    for k, ai in enumerate(assistant_idx):
        # 以第 ai 轮 assistant 回复为监督目标
        l, r = 0, ai
        cur = convs[l:r+1]

        # 试编码：如果长度已不超，直接收下
        data_dict = text_preprocess(copy.deepcopy(cur))
        ids = data_dict["input_ids"]
        if len(ids) <= max_len:
            new_item = dict(sample)
            new_item["conversations"] = cur
            # 给展开后的样本一个可区分的 id
            if base_id is not None:
                new_item["id"] = f"{base_id}#a{k:02d}"
            out.append(new_item)
            continue

        # 过长：从左侧逐步删除历史 turn，直至长度 <= max_len
        while l < r:
            l += 1
            cur = convs[l:r+1]
            data_dict = text_preprocess(copy.deepcopy(cur))
            ids = data_dict["input_ids"]
            if len(ids) <= max_len:
                break

        # 若仍然超长，说明“目标这轮回复本身+模板”就超过 max_len，只能报错
        if len(ids) > max_len:
            raise RuntimeError(
                f"[EXPAND] Single assistant reply exceeds model_max_length "
                f"(len={len(ids)} > {max_len}). Increase context length or shorten templates."
            )

        new_item = dict(sample)
        new_item["conversations"] = cur
        if base_id is not None:
            new_item["id"] = f"{base_id}#a{k:02d}"
        out.append(new_item)

    return out



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: "DataArguments"):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

        # --- turn-level expansion (no truncation on supervised target) ---
        max_len = tokenizer.model_max_length
        expanded = []
        for s in list_data_dict:
            if isinstance(s.get("conversations", None), list) and len(s["conversations"]) > 2:
                expanded.extend(_split_multiturn_into_samples(s, self.text_preprocess, tokenizer, max_len))
            else:
                expanded.append(s)

        self.list_data_dict = expanded
        print(f"[EXPAND] expanded samples: {len(expanded)} (from {len(list_data_dict)})")

        # ---- DEBUG: 只在主进程打印单/多轮分布 ----
        def _is_main_process():
            return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"

        if _is_main_process():
            dist = Counter()
            examples = {"single": [], "multi": []}
            for i, s in enumerate(self.list_data_dict):
                convs = s.get("conversations")
                if convs is None:
                    convs = s.get("turns", [])
                roles = []
                for c in convs:
                    r = c.get("role") or c.get("from")
                    if r is None and "human" in c:      r = "human"
                    elif r is None and "assistant" in c: r = "assistant"
                    roles.append(r)
                user_turns = sum(1 for r in roles if r in ("user", "human"))
                key = "multi" if user_turns > 1 else "single"
                dist[key] += 1
                if len(examples[key]) < 3:
                    examples[key].append(s.get("id", i))
            print(
                f"[SANITY][DATASET] single/multi={dict(dist)}; "
                f"sample_ids(single)={examples['single']}, sample_ids(multi)={examples['multi']}"
            )

    def __len__(self):
        return len(self.list_data_dict)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = -100   # ← 新增：统一用实例属性，不再用局部/全局名

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=self.ignore_index)
        
        
        assert input_ids.size(1) <= self.tokenizer.model_max_length, \
            f"sequence too long: {input_ids.size(1)} > {self.tokenizer.model_max_length}"
        assert labels.size(1) <= self.tokenizer.model_max_length, \
            f"labels too long: {labels.size(1)} > {self.tokenizer.model_max_length}"
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
       
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask)
        
        if "image" in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
                
        if not hasattr(self, "_dbg_tok"):
            self._dbg_tok = 0
        if self._dbg_tok < 3:
            
            lab = batch["labels"]
            train_tok = (lab != self.ignore_index).sum(dim=1).tolist()
            print("[SANITY] trainable_tokens_per_sample:", train_tok[:8],
                "seq_len=", lab.size(1))
            self._dbg_tok += 1


        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
