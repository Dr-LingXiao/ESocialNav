import os
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from .tinyllava_trainer import LLaVATrainer
from ..utils.constants import IGNORE_INDEX


class LLaVADPOTrainer(LLaVATrainer):
    """
    设计原则：
    - 训练主链路完全复用父类 LLaVATrainer（SFT）逻辑；
    - 仅在 compute_loss 中“按概率”追加一次 DPO 损失；
    - dpo_* 相关参数/loader 由子类持有，绝不往父类 __init__ 传。
    """

    def __init__(self, *args, **kwargs):
        # ---- 截获并保存 DPO 相关参数（不要传给父类） ----
        self._dpo_loader = kwargs.pop("dpo_dataloader", None)
        self.ref_model: Optional[torch.nn.Module] = kwargs.pop("ref_model", None)

        # 这些是训练脚本传入的可选参数（没有就给默认）
        self.dpo_prob: float = float(kwargs.pop("dpo_prob", 0.0) or 0.0)
        self.dpo_weight: float = float(kwargs.pop("dpo_weight", 1.0))
        self.dpo_beta: float = float(kwargs.pop("dpo_beta", 0.1))
        # 下面这几个目前没在本文件里用到，但保留下来以兼容你的调用
        self.dpo_margin: float = float(kwargs.pop("dpo_margin", 0.0))
        self.dpo_label_smoothing: float = float(kwargs.pop("dpo_label_smoothing", 0.0))
        self.dpo_ipo: bool = bool(kwargs.pop("dpo_ipo", False))

        # 父类初始化（只保留父类认识的参数）
        super().__init__(*args, **kwargs)

        # 内部状态
        self._dpo_iter = None  # 惰性创建
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

    # -------------------- DPO 辅助 --------------------

    def _next_dpo_batch(self) -> Optional[Dict[str, Any]]:
        if self._dpo_loader is None or self.dpo_prob <= 0.0:
            return None
        if self._dpo_iter is None:
            self._dpo_iter = iter(self._dpo_loader)
        try:
            batch = next(self._dpo_iter)
        except StopIteration:
            self._dpo_iter = iter(self._dpo_loader)
            batch = next(self._dpo_iter)
        return batch

    def _pack_images(self, d: Dict[str, Any]) -> Optional[torch.Tensor]:
        # 支持常见命名：images / image / pixel_values
        if "images" in d and d["images"] is not None:
            return d["images"]
        if "image" in d and d["image"] is not None:
            return d["image"]
        if "pixel_values" in d and d["pixel_values"] is not None:
            return d["pixel_values"]
        return None

    def _logps(self, model: torch.nn.Module,
           input_ids: torch.Tensor,
           attention_mask: torch.Tensor,
           images: Optional[torch.Tensor],
           labels: torch.Tensor) -> torch.Tensor:
        """
        计算每样本 token 对数似然（忽略 IGNORE_INDEX）。
        关键：TinyLLaVA 只有在收到 images （且很多实现也依赖 labels）时才会做多模态展开；
            否则会把负的 <image> 哨兵当成普通 token 送进 embed_tokens 而崩掉。
        """
        # —— 先做友好断言，避免再掉进 CUDA 黑盒 —— 
        try:
            from tinyllava.utils.constants import IMAGE_TOKEN_INDEX
        except Exception:
            IMAGE_TOKEN_INDEX = -200

        if (input_ids < 0).any():
            # 允许的唯一负值是 IMAGE_TOKEN_INDEX；否则直接报非法
            bad = input_ids[(input_ids < 0) & (input_ids != IMAGE_TOKEN_INDEX)]
            if bad.numel() > 0:
                raise RuntimeError(f"[DPO/_logps] 非法负 token id: {torch.unique(bad)}")
            if images is None:
                raise RuntimeError("[DPO/_logps] input_ids 含 <image> 哨兵，但 images=None；"
                                "请检查 collator→trainer→model 的 'images' 键是否丢失/改名。")

        # —— 关键：同时传 labels 和 images，强制走多模态路径 —— 
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "use_cache": False,
        }
        outputs = model(**kwargs)                  # 现在会由 TinyLLaVA 做 <image> 展开
        logits = outputs.logits                    # [B, T, V]

        # 取真实标签的 log-prob
        logprobs = torch.log_softmax(logits, dim=-1)                     # [B,T,V]
        gather   = logprobs.gather(dim=-1, index=labels.unsqueeze(-1))   # [B,T,1]
        token_lp = gather.squeeze(-1)                                    # [B,T]

        from ..utils.constants import IGNORE_INDEX
        mask  = (labels != IGNORE_INDEX).to(token_lp.dtype)              # [B,T]
        denom = mask.sum(dim=-1).clamp_min(1.0)                          # [B]
        return (token_lp * mask).sum(dim=-1) / denom                     # [B]

    



    def _compute_dpo_loss(self, dpo_batch):
        """
        同时兼容两种 batch 结构：
        A) 扁平键（collator 输出）：
        {
            "input_ids_chosen": Tensor[B,L], "attention_mask_chosen": Tensor[B,L], "labels_chosen": Tensor[B,L],
            "input_ids_rejected": ..., "attention_mask_rejected": ..., "labels_rejected": ...,
            "images": Tensor[B,C,H,W] (可选)
        }
        B) 多轮列表：
        {
            "chosen":{"input_ids":[T], "attention_mask":[T], "labels":[T]},
            "rejected":{"input_ids":[T], "attention_mask":[T], "labels":[T]},
            "images": Tensor[B,C,H,W] (可选)
        }
        """
        import torch
        import torch.nn.functional as F

        images = dpo_batch.get("images", None)

        # ---- 归一化为“多轮列表”结构 ----
        if "chosen" in dpo_batch and isinstance(dpo_batch["chosen"], dict):
            # 已是多轮列表结构
            chosen = dpo_batch["chosen"]
            rejected = dpo_batch["rejected"]
        else:
            # 扁平键 → 包装成单轮列表（T=1）
            chosen = {
                "input_ids":      [dpo_batch["input_ids_chosen"]],
                "attention_mask": [dpo_batch["attention_mask_chosen"]],
                "labels":         [dpo_batch["labels_chosen"]],
            }
            rejected = {
                "input_ids":      [dpo_batch["input_ids_rejected"]],
                "attention_mask": [dpo_batch["attention_mask_rejected"]],
                "labels":         [dpo_batch["labels_rejected"]],
            }

        c_list = list(zip(chosen["input_ids"], chosen["attention_mask"], chosen["labels"]))
        r_list = list(zip(rejected["input_ids"], rejected["attention_mask"], rejected["labels"]))
        assert len(c_list) == len(r_list), "chosen/rejected 轮数不一致"
        T = len(c_list)
        if T == 0:
            raise ValueError("空的多轮 batch")

        device_pi = self.model.device
        device_ref = self.ref_model.device if getattr(self, "ref_model", None) is not None else None

        images_pi = images.to(device_pi) if (images is not None and isinstance(images, torch.Tensor)) else None
        images_ref = images.to(device_ref) if (images_pi is not None and device_ref is not None) else None

        pi_c_all, pi_r_all = [], []
        ref_c_all, ref_r_all = [], []

        for t in range(T):
            ci, ca, cl = c_list[t]
            ri, ra, rl = r_list[t]

            ci, ca, cl = ci.to(device_pi), ca.to(device_pi), cl.to(device_pi)
            ri, ra, rl = ri.to(device_pi), ra.to(device_pi), rl.to(device_pi)

            # 统一走 self._logps（内部已处理 IGNORE_INDEX / images 多模态）
            c_lp_pi = self._logps(self.model, ci, ca, images_pi, cl)  # [B]
            r_lp_pi = self._logps(self.model, ri, ra, images_pi, rl)  # [B]
            pi_c_all.append(c_lp_pi)
            pi_r_all.append(r_lp_pi)

            if self.ref_model is not None:
                ci_r, ca_r, cl_r = ci.to(device_ref), ca.to(device_ref), cl.to(device_ref)
                ri_r, ra_r, rl_r = ri.to(device_ref), ra.to(device_ref), rl.to(device_ref)
                c_lp_ref = self._logps(self.ref_model, ci_r, ca_r, images_ref, cl_r)
                r_lp_ref = self._logps(self.ref_model, ri_r, ra_r, images_ref, rl_r)
                ref_c_all.append(c_lp_ref)
                ref_r_all.append(r_lp_ref)

        pi_c = torch.stack(pi_c_all, dim=1)  # [B, T]
        pi_r = torch.stack(pi_r_all, dim=1)  # [B, T]

        if self.ref_model is None:
            diff = pi_c - pi_r
            dpo_loss = -torch.logsigmoid(self.dpo_beta * diff).mean()
        else:
            ref_c = torch.stack(ref_c_all, dim=1)
            ref_r = torch.stack(ref_r_all, dim=1)
            logits = (pi_c - pi_r) - (ref_c - ref_r)
            dpo_loss = -torch.logsigmoid(self.dpo_beta * logits).mean()

        return dpo_loss




    # -------------------- 与父类对齐的 compute_loss --------------------

    def _to_sft_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        若主 DataLoader 恰好传入了 DPO 扁平键，转为 SFT 标准键；否则原样返回。
        """
        if "input_ids" in inputs and ("labels" in inputs or "attention_mask" in inputs):
            return inputs

        if "chosen" in inputs and isinstance(inputs["chosen"], dict):
            return inputs["chosen"]

        sft = {}
        for k, v in inputs.items():
            if k.endswith("_chosen"):
                base = k[:-7]
                if base == "pixel_values":
                    base = "images"
                sft[base] = v
        if sft:
            if "images" not in sft:
                if "pixel_values" in sft:
                    sft["images"] = sft.pop("pixel_values")
                elif "images_chosen" in inputs:
                    sft["images"] = inputs["images_chosen"]
                elif "images" in inputs:
                    sft["images"] = inputs["images"]  # ★ 新增：从顶层补回
            return sft

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        两个保证：
        1) 未启用 DPO（prob<=0 或无 dpo_loader 或 eval）→ 严格走父类 SFT 路径，return_outputs 语义与 train.py 完全一致；
        2) 启用 DPO → 在 SFT 的 loss 基础上按概率追加 DPO。
        """
        inputs_sft = self._to_sft_inputs(inputs)
        
        
        

        # --- 情况A：纯 SFT（与 train.py 完全一致）---
        dpo_disabled = (self.dpo_prob <= 0.0) or (self._dpo_loader is None) or (not getattr(model, "training", False))
        if dpo_disabled:
            return super().compute_loss(model, inputs_sft, return_outputs)

        # --- 情况B：SFT + 按概率追加 DPO ---
        
         # —— SFT 前置自检：含负 token 必须带 images —— 
        try:
            from tinyllava.utils.constants import IMAGE_TOKEN_INDEX
        except Exception:
            IMAGE_TOKEN_INDEX = -200
        if "input_ids" in inputs_sft and isinstance(inputs_sft["input_ids"], torch.Tensor):
            ids = inputs_sft["input_ids"]
            if (ids < 0).any() and (("images" not in inputs_sft) or (inputs_sft["images"] is None)):
                negs = torch.unique(ids[ids < 0]).tolist()
                raise RuntimeError(f"[SFT] input_ids 含负 token {negs}，但缺少 images；"
                                   f"请检查 _to_sft_inputs 是否把顶层 images 带进来了。")
                
        sft_loss, sft_outputs = super().compute_loss(model, inputs_sft, return_outputs=True)
        total_loss = sft_loss

        if torch.rand(1).item() < self.dpo_prob:
            dpo_batch = self._next_dpo_batch()
            if dpo_batch is not None:
                dpo_loss = self._compute_dpo_loss(dpo_batch)
                total_loss = total_loss + self.dpo_weight * dpo_loss

        return (total_loss, sft_outputs) if return_outputs else total_loss
