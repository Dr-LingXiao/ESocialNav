import copy, json, os, random 
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, List, Tuple

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import transformers

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import IGNORE_INDEX
from .dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset  # SFT 管线
# ---------------------- DPO 数据集 ----------------------
# dataset_dpo.py 顶部合适位置（imports 附近）


class LazyDPODialogueDataset(Dataset):
    """
    支持两种配对模式：
      - 'last'（默认，推荐）：每段对话只取“最后一轮”做 DPO（与 SFT 粒度一致：1 段 = 1 样本）
      - 'all'：逐轮展开（历史 0..t-1 + 当前 t 做比较），N 轮对话变 N 个样本（旧行为）
      - 'random'：每段对话随机选一轮做比较（长程统计≈1 段 = 1 样本）
    配置来源优先级：data_args.dpo_pair_mode -> 环境变量 DPO_PAIR_MODE -> 默认 'last'
    可选裁剪：data_args.dpo_max_history_turns（仅保留最近 K 轮历史）
    """

    def __init__(self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments
    ):
        super().__init__()
        assert isinstance(data_path, str) and os.path.exists(data_path), f"data_path 不存在: {data_path}"
        self.dialogues = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

        # 读取配对模式
        self.pair_mode = str(getattr(self.data_args, "dpo_pair_mode", os.getenv("DPO_PAIR_MODE", "last"))).lower()
        assert self.pair_mode in ("last", "all", "random"), f"dpo_pair_mode 不支持: {self.pair_mode}"

        # 建索引：
        #  - 'all'：逐轮展开 -> (di, ti)
        #  - 其它：每段取 1 条 -> (di, -1)，具体选哪一轮在 __getitem__ 决定（last 或 random）
        self.index: List[Tuple[int, int]] = []
        for di, dlg in enumerate(self.dialogues):
            turns = dlg.get("turns", [])
            if len(turns) == 0:
                continue
            if self.pair_mode == "all":
                for ti in range(len(turns)):
                    self.index.append((di, ti))
            else:
                self.index.append((di, -1))

    def __len__(self):
        return len(self.index)

    # —— 工具：确保整段对话只在“首条 human”放一次 <image>，与 SFT 管线对齐 ——
    def _build_conversations(self, history_pairs, cur_human, cur_answer, with_image: bool):
        conv = []
        inserted = False  # 是否已放过 <image>

        def _maybe_add_image(q: str) -> str:
            nonlocal inserted
            if not with_image:
                return q
            # 去重：先移除已有的 <image>（避免重复）
            q = q.replace("<image>\n", "").replace("<image>", "")
            if not inserted:
                inserted = True
                return "<image>\n" + q
            return q

        # 历史 (human -> chosen)
        for hq, ha in history_pairs:
            hq = _maybe_add_image(hq)
            conv.append({"from": "human", "value": hq})
            conv.append({"from": "gpt",   "value": ha})

        # 当前轮 (human -> cur_answer)
        cur_human = _maybe_add_image(cur_human)
        conv.append({"from": "human", "value": cur_human})
        conv.append({"from": "gpt",   "value": cur_answer})
        return conv

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        di, ti_idx = self.index[i]
        dlg = self.dialogues[di]
        turns = dlg.get("turns", [])
        assert len(turns) > 0, "空 turns"

        # 选择当前比较的轮次：all 模式用索引；last 用最后一轮；random 随机一轮
        if self.pair_mode == "all":
            t_use = ti_idx
        elif self.pair_mode == "last":
            t_use = len(turns) - 1
        else:  # random
            t_use = random.randrange(len(turns))

        # 取历史 0..t_use-1 轮（用 chosen 作为上一轮回答）
        def _get_human(t):
            return t.get("human", t.get("prompt", t.get("promot", "")))

        history_pairs = [(_get_human(turns[k]), turns[k]["chosen"]) for k in range(t_use)]

        # 可选裁剪：仅保留最近 K 轮历史
        k = getattr(self.data_args, "dpo_max_history_turns", None)
        if isinstance(k, int) and k > 0:
            history_pairs = history_pairs[-k:]

        # 当前轮的人类问题 & 两个候选回答
        cur_h = _get_human(turns[t_use])
        ans_c = turns[t_use]["chosen"]
        ans_r = turns[t_use]["rejected"]

        with_image = bool(dlg.get("image"))

        # 组两份 conversations：历史 + (human -> chosen) / (human -> rejected)
        conv_c = self._build_conversations(history_pairs, cur_h, ans_c, with_image)
        conv_r = self._build_conversations(history_pairs, cur_h, ans_r, with_image)

        # 文本编码
        data_c = self.text_preprocess(copy.deepcopy(conv_c))
        data_r = self.text_preprocess(copy.deepcopy(conv_r))

        out = {
            "input_ids_chosen":   data_c["input_ids"],
            "labels_chosen":      data_c["labels"],
            "input_ids_rejected": data_r["input_ids"],
            "labels_rejected":    data_r["labels"],
        }

        # 图像处理（与 SFT 保持一致：若是多模态但该样本没图，给占位零张）
        
        # 图像处理（与 SFT 保持一致：若是多模态但该样本没图，给占位零张）
        if with_image:
            img_path = os.path.join(self.data_args.image_folder, dlg["image"])
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                crop_size = getattr(self.data_args.image_processor, 'crop_size',
                                    getattr(self.data_args.image_processor, 'size'))
                image = torch.zeros(3, crop_size['height'], crop_size['width'])
                print(f"[WARN][DPO] fail to open {img_path}: {e}; use zeros.", flush=True)
            out["image"] = self.image_preprocess(image)
        elif getattr(self.data_args, "is_multimodal", False):
            crop_size = getattr(self.data_args.image_processor, 'crop_size',
                                getattr(self.data_args.image_processor, 'size'))
            out["image"] = torch.zeros(3, crop_size['height'], crop_size['width'])


        return out


@dataclass
class DataCollatorForDPODialogue:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        def _pad(seqs, pad_id):
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)

        pad_id = self.tokenizer.pad_token_id
        max_len = self.tokenizer.model_max_length

        ids_c = _pad([e["input_ids_chosen"]   for e in instances], pad_id)
        ids_r = _pad([e["input_ids_rejected"] for e in instances], pad_id)
        lab_c = _pad([e["labels_chosen"]      for e in instances], IGNORE_INDEX)
        lab_r = _pad([e["labels_rejected"]    for e in instances], IGNORE_INDEX)

        # ✅ 保尾截断（确保“最后一轮回答”的监督在内）
        if ids_c.size(1) > max_len:
            ids_c = ids_c[:, -max_len:]
            lab_c = lab_c[:, -max_len:]
        if ids_r.size(1) > max_len:
            ids_r = ids_r[:, -max_len:]
            lab_r = lab_r[:, -max_len:]

        # 现在再按截断后的序列重建 mask
        att_c = ids_c.ne(pad_id).long()
        att_r = ids_r.ne(pad_id).long()

        # 在 collator 里，加载并堆叠图片后：
        
                # —— 组 batch（文本）——
        batch = {
            "input_ids_chosen": ids_c,
            "labels_chosen": lab_c,
            "attention_mask_chosen": att_c,
            "input_ids_rejected": ids_r,
            "labels_rejected": lab_r,
            "attention_mask_rejected": att_r,
        }

        # —— 组 batch（图像，来自 __getitem__ 里的 'image'）——
        if "image" in instances[0]:
            images = [inst["image"] for inst in instances]
            # 尽量 stack 成 [B,C,H,W]；若尺寸不一，直接列表交给后续图片处理（通常不建议）
            can_stack = all((img is not None) and (img.shape == images[0].shape) for img in images)
            batch["images"] = torch.stack(images) if can_stack else images


        # 如果你还有“附带的 SFT 视图”，也同样保尾截断（否则 CE 也可能全是 -100）
        if "sft_input_ids" in instances[0]:
            sft_ids = _pad([e["sft_input_ids"] for e in instances], pad_id)
            sft_lab = _pad([e["sft_labels"]    for e in instances], IGNORE_INDEX)
            if sft_ids.size(1) > max_len:
                sft_ids = sft_ids[:, -max_len:]
                sft_lab = sft_lab[:, -max_len:]
            sft_att = sft_ids.ne(pad_id).long()
            batch["sft_input_ids"] = sft_ids
            batch["sft_attention_mask"] = sft_att
            batch["sft_labels"] = sft_lab
            
            
        # --- DEBUG: 快速发现“非法负 id”或越界 id（允许 IMAGE_TOKEN_INDEX） ---
        vsz = len(self.tokenizer)
        try:
            from tinyllava.utils.constants import IMAGE_TOKEN_INDEX
        except Exception:
            IMAGE_TOKEN_INDEX = -200  # 兜底

        def _chk_allow_image_token(t):
            if t.numel() == 0:
                return
            # 允许 IMAGE_TOKEN_INDEX，其他负数一律报错
            bad = (t < 0) & (t != IMAGE_TOKEN_INDEX)
            if bad.any():
                # 列出所有非法负数，便于定位
                bad_vals = torch.unique(t[bad]).tolist()
                raise RuntimeError(
                    f"[BAD INPUT IDS] negatives other than IMAGE_TOKEN_INDEX "
                    f"({IMAGE_TOKEN_INDEX}) found: {bad_vals}"
                )
            # 词表上界检查
            mx = int(t.max().item())
            if mx >= vsz:
                raise RuntimeError(f"[OOB INPUT IDS] max input id {mx} >= vocab size {vsz}")

        _chk_allow_image_token(batch["input_ids_chosen"])
        _chk_allow_image_token(batch["input_ids_rejected"])

      
        return batch



# ---------------------- 混合：联合数据集 + Collator + 比例 BatchSampler ----------------------
class MixedTaggedDataset(Dataset):
    """把若干 SFT 与 DPO 数据集合并为一个联合数据集；索引空间：[0,sft_len)为 SFT，[sft_len, sft_len+dpo_len) 为 DPO。"""
    def __init__(self, sft_list: List[Dataset], dpo_list: List[Dataset]):
        assert len(sft_list) > 0 and len(dpo_list) > 0, "需要同时包含 SFT 与 DPO 数据集"
        self.sft = sft_list[0] if len(sft_list) == 1 else ConcatDataset(sft_list)
        self.dpo = dpo_list[0] if len(dpo_list) == 1 else ConcatDataset(dpo_list)
        self.sft_len = len(self.sft)
        self.dpo_len = len(self.dpo)

    def __len__(self):
        return self.sft_len + self.dpo_len

    def __getitem__(self, idx):
        if idx < self.sft_len:
            item = self.sft[idx]
            item["__kind__"] = "sft"
            return item
        else:
            j = idx - self.sft_len
            item = self.dpo[j]
            item["__kind__"] = "dpo"
            return item


class MixedAutoCollator:
    """
    批内自动分组：
      - 仅 SFT：DataCollatorForSupervisedDataset
      - 仅 DPO：DataCollatorForDPODialogue
      - 混合：分别打包后合并；避免 'images' 冲突，改名为 images_sft / images_dpo
    """
    def __init__(self, tokenizer):
        self.sft = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        self.dpo = DataCollatorForDPODialogue(tokenizer=tokenizer)
        
        
    def __call__(self, instances):
        sft_items = [x for x in instances if x.get("__kind__") == "sft"]
        dpo_items = [x for x in instances if x.get("__kind__") == "dpo"]
        if len(dpo_items) == 0:
            return self.sft(sft_items)
        if len(sft_items) == 0:
            return self.dpo(dpo_items)

        batch_sft = self.sft(sft_items)
        batch_dpo = self.dpo(dpo_items)

        # 避免键冲突，先改名
        if "images" in batch_sft:
            batch_sft["images_sft"] = batch_sft.pop("images")
        if "images" in batch_dpo:
            batch_dpo["images_dpo"] = batch_dpo.pop("images")

        # 合并
        batch = {}
        batch.update(batch_sft)
        batch.update(batch_dpo)

        # ✅ 关键：给 DPO 的图再挂一个通用别名，保证 Trainer 按 'images' 能拿到
        if "images_dpo" in batch and "images" not in batch:
            batch["images"] = batch["images_dpo"]

        return batch

  


class RatioDistributedBatchSampler(Sampler[List[int]]):
    """
    自定义比例（每批 SFT:DPO = sft_ratio : 1-sft_ratio）的分布式 BatchSampler。
    - batch_size 任意整数；k_sft = round(batch_size * sft_ratio)，k_dpo = batch_size - k_sft。
    - 若某侧样本不足以按比例组满混合批，则自动退化为“单侧批”（只 SFT 或只 DPO）以保证有数据能训练。
    - world_size/rank：全局批序列按轮转法切分，第 b 个批归 (b % world_size) 号 rank。
    """
    
    def __init__(self,
                 sft_len: int, dpo_len: int,
                 sft_offset: int, dpo_offset: int,
                 batch_size: int,
                 sft_ratio: float = 0.5,
                 seed: int = 42,
                 world_size: int = 1,
                 rank: int = 0,
                 drop_last: bool = True):
        assert 0 <= rank < world_size
        self.sft_len = int(sft_len)
        self.dpo_len = int(dpo_len)
        self.sft_offset = int(sft_offset)
        self.dpo_offset = int(dpo_offset)
        self.bs = int(max(1, batch_size))
        self.r = float(max(0.0, min(1.0, sft_ratio)))
        self.seed = int(seed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)

        # 统一乱序（全局一致）
        g = torch.Generator()
        g.manual_seed(self.seed)
        sft_perm = torch.randperm(self.sft_len, generator=g).tolist()
        dpo_perm = torch.randperm(self.dpo_len, generator=g).tolist()

        # 目标：k_sft 在 {floor, ceil} 间切换，使得长程统计接近 bs * ratio
        k_floor = int(self.bs * self.r // 1)                # floor
        frac    = (self.bs * self.r) - k_floor              # 小数部分 in [0,1)
        carry   = 0.0

        ptr_s, ptr_d = 0, 0
        batches: List[List[int]] = []

        # 连续生成“混合批”（两侧都要能满足）
        while True:
            # 本批 k_sft：floor + 误差累加到 1 就补 1
            k_sft = k_floor
            if 0.0 < self.r < 1.0 and self.bs >= 2:
                carry += frac
                if carry >= 1.0:
                    k_sft += 1
                    carry -= 1.0
                # 防守：至少 1、至多 bs-1
                k_sft = max(1, min(self.bs - 1, k_sft))
            elif self.r == 0.0:
                k_sft = 0
            else:  # self.r == 1.0
                k_sft = self.bs

            k_dpo = self.bs - k_sft

            # 检查当批是否有足够样本
            if (ptr_s + k_sft) > self.sft_len or (ptr_d + k_dpo) > self.dpo_len:
                break

            s_blk = sft_perm[ptr_s:ptr_s + k_sft]; ptr_s += k_sft
            d_blk = dpo_perm[ptr_d:ptr_d + k_dpo]; ptr_d += k_dpo

            batch = [self.sft_offset + i for i in s_blk] + [self.dpo_offset + i for i in d_blk]
            batches.append(batch)

        # 如果本轮没有混合批拼出来，则退化为单侧（尽量不空转）
        if len(batches) == 0:
            if self.r == 0.0 and self.dpo_len >= self.bs:      # 全 DPO
                n = self.dpo_len // self.bs
                for b in range(n):
                    blk = dpo_perm[b*self.bs:(b+1)*self.bs]
                    batches.append([self.dpo_offset + i for i in blk])
            elif self.r == 1.0 and self.sft_len >= self.bs:     # 全 SFT
                n = self.sft_len // self.bs
                for b in range(n):
                    blk = sft_perm[b*self.bs:(b+1)*self.bs]
                    batches.append([self.sft_offset + i for i in blk])
            else:
                # 比例非 0/1，但两侧之一太少：如果不 drop_last，可以把能组满的一侧也拼上
                if not self.drop_last:
                    if self.sft_len - ptr_s >= self.bs:
                        n = (self.sft_len - ptr_s) // self.bs
                        for b in range(n):
                            blk = sft_perm[ptr_s + b*self.bs : ptr_s + (b+1)*self.bs]
                            batches.append([self.sft_offset + i for i in blk])
                    if self.dpo_len - ptr_d >= self.bs:
                        n = (self.dpo_len - ptr_d) // self.bs
                        for b in range(n):
                            blk = dpo_perm[ptr_d + b*self.bs : ptr_d + (b+1)*self.bs]
                            batches.append([self.dpo_offset + i for i in blk])

        # 分布式切分
        # …… RatioDistributedBatchSampler.__init__ 的末尾，分布式切分之后
        self.global_batches = [b for i, b in enumerate(batches) if (i % self.world_size) == self.rank]

        # ✅ 仅 rank0 打印调试信息（可选）
        if self.rank == 0:
            print(
                f"[mix] bs={self.bs} ratio={self.r:.3f} batches={len(batches)} "
                f"(sft_len={self.sft_len}, dpo_len={self.dpo_len})",
                flush=True
            )

    def __iter__(self):
        for batch in self.global_batches:
            yield batch

    def __len__(self):
        return len(self.global_batches)
    
    


# ---------------------- 构建数据模块 ----------------------
def _split_paths(s: str) -> List[str]:
    parts = [p.strip() for p in s.replace("|", ",").split(",") if p.strip()]
    return parts

def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """
    支持：
      - 仅 SFT：返回 SFT 数据集 + SFT collator
      - 仅 DPO：返回 DPO 数据集 + DPO collator
      - SFT + DPO：返回 MixedTaggedDataset + MixedAutoCollator（Trainer 内部用 RatioDistributedBatchSampler 采样）
    """
    paths = _split_paths(data_args.data_path)
    assert len(paths) >= 1, "data_path 不能为空"

    sft_list: List[Dataset] = []
    dpo_list: List[Dataset] = []

    for p in paths:
        assert os.path.exists(p), f"数据文件不存在: {p}"
        peek = json.load(open(p, "r"))
        is_sft = isinstance(peek, list) and len(peek) > 0 and isinstance(peek[0], dict) and ("conversations" in peek[0])
        if is_sft:
            ds = LazySupervisedDataset(tokenizer=tokenizer, data_path=p, data_args=data_args)
            sft_list.append(ds)
        else:
            ds = LazyDPODialogueDataset(tokenizer=tokenizer, data_path=p, data_args=data_args)
            dpo_list.append(ds)

    # 只 SFT
    if len(dpo_list) == 0 and len(sft_list) > 0:
        train_dataset = sft_list[0] if len(sft_list) == 1 else ConcatDataset(sft_list)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    # 只 DPO
    if len(sft_list) == 0 and len(dpo_list) > 0:
        train_dataset = dpo_list[0] if len(dpo_list) == 1 else ConcatDataset(dpo_list)
        data_collator = DataCollatorForDPODialogue(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    # 混合：联合数据集 + 自动 collator（采样器在 Trainer 里构造）
    mixed = MixedTaggedDataset(sft_list=sft_list, dpo_list=dpo_list)
    collator = MixedAutoCollator(tokenizer=tokenizer)
    return dict(train_dataset=mixed, eval_dataset=None, data_collator=collator)
