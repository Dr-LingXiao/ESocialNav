# tinyllava/train/mix_utils.py
from dataclasses import dataclass

@dataclass
class DPOCollatorWithSFTView:
    """
    在 DPO collator 之上，额外生成 SFT 所需字段（用 chosen 分支当 SFT）。
    纯文本/多模态都支持；若没有图像字段则不添加 images。
    """
    base_collator: object
    tokenizer: object  # 只需要 pad_token_id

    def __call__(self, features):
        batch = self.base_collator(features)  # 标准 DPO 批：*_chosen / *_rejected / 可能 images_*
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)

        # 只有真的拿到了 DPO 的 chosen 才生成 SFT 视图
        if ("input_ids_chosen" in batch) and ("labels_chosen" in batch):
            batch["input_ids"] = batch["input_ids_chosen"]
            batch["labels"] = batch["labels_chosen"]

            # 优先使用已提供的 mask；否则根据 pad 推断
            if "attention_mask_chosen" in batch:
                batch["attention_mask"] = batch["attention_mask_chosen"]
            else:
                batch["attention_mask"] = (batch["input_ids"] != pad_id).long()

            # 多模态：若有分支图像，则复用给 SFT 视图
            if "images_chosen" in batch:
                batch["images"] = batch["images_chosen"]

        return batch


def wrap_dpo_dm_with_sft_view(dpo_dm: dict, tokenizer):
    """
    输入：dpo_dm = {'train_dataset': ..., 'eval_dataset': ..., 'data_collator': ...}
    输出：同样结构的 dict，但 data_collator 被包装成能同时产出 SFT 视图。
    """
    new_dm = dict(dpo_dm)  # 浅拷贝避免副作用
    new_dm["data_collator"] = DPOCollatorWithSFTView(dpo_dm["data_collator"], tokenizer)
    return new_dm
