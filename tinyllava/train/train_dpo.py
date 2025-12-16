from packaging import version
import os
from torch.utils.data import DataLoader, Subset, Dataset
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
import transformers
import torch

from tinyllava.utils import *
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.data.dataset_dpo import make_dpo_data_module
from tinyllava.train.tinyllava_trainer_dpo import LLaVADPOTrainer
from tinyllava.utils.arguments import ModelArguments, DataArguments, TrainingArguments
from tinyllava.data.dataset_dpo import DataCollatorForDPODialogue
# ==== BEGIN: crash_log_hook ====
# ==== BEGIN: force error log to fixed path ====
import sys, faulthandler, traceback

_LOG_PATH = "/home/ling/TinyLLaVA_Factory/error_full.log"
_ERR_FH = open(_LOG_PATH, "w", buffering=1, encoding="utf-8")

# 打开 faulthandler，能抓住 segfault/cuda assert 的 Python 栈
faulthandler.enable(_ERR_FH)

def _excepthook(exctype, value, tb):
    print("\n========== UNCAUGHT EXCEPTION ==========", file=_ERR_FH, flush=True)
    traceback.print_exception(exctype, value, tb, file=_ERR_FH)
    print("========================================", file=_ERR_FH, flush=True)
    # 同时在控制台也输出，方便调试
    traceback.print_exception(exctype, value, tb)

sys.excepthook = _excepthook

print(f"[CRASHLOG] Logging all errors to {_LOG_PATH}", file=_ERR_FH, flush=True)
# ==== END: force error log to fixed path ====

# ==== END: crash_log_hook ====



def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio
    return {
        "llm": {
            "model_name_or_path": model_arguments.model_name_or_path,
            "cache_dir": model_arguments.cache_dir,
            "attn_implementation": model_arguments.attn_implementation,
        },
        "vision_tower": {
            "model_name_or_path": model_arguments.vision_tower.split(":")[-1],
            **({"model_name_or_path2": model_arguments.vision_tower2.split(":")[-1]} if model_arguments.vision_tower2 else {}),
        },
        "connector": {"connector_type": model_arguments.connector_type},
    }

def _load_tinyllava_from_recipe(training_recipe, model_args, base_cfg, pretrained_root: str | None):
    model = TinyLlavaForConditionalGeneration(base_cfg)
    if pretrained_root:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args["llm"])
        model.load_vision_tower(**model_args["vision_tower"])
        model.load_connector(**model_args["connector"])
    return model

class ProgressCallback(TrainerCallback):
    def __init__(self, output_dir: str | None):
        self.output_dir = output_dir
    def _touch(self, name: str):
        if not self.output_dir: return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            open(os.path.join(self.output_dir, f"{name}.txt"), "a").close()
        except Exception:
            pass
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._touch("on_train_begin")

def _split_sft_dpo_from_mixed(ds: Dataset):
    """
    若 make_dpo_data_module 未显式返回 sft/dpo 子数据集，
    尝试从 MixedTaggedDataset 属性拆分（[0:sft_len) SFT， [sft_len:sft_len+dpo_len) DPO）。
    """
    sft_ds = None
    dpo_ds = None
    if hasattr(ds, "sft_len") and hasattr(ds, "dpo_len"):
        sft_len = int(ds.sft_len)
        dpo_len = int(ds.dpo_len)
        if sft_len > 0:
            sft_ds = Subset(ds, range(0, sft_len))
        if dpo_len > 0:
            dpo_ds = Subset(ds, range(sft_len, sft_len + dpo_len))
    return sft_ds, dpo_ds

def _int_from_env(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "")
        return int(v) if v.strip() else default
    except Exception:
        return default

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    print(
        f"[ARGS] conv_version={getattr(data_arguments, 'conv_version', None)}  "
        f"data_path={getattr(data_arguments, 'data_path', None)}  "
        f"sft_data_path={getattr(data_arguments, 'sft_data_path', None)}  "
        f"dpo_data_path={getattr(data_arguments, 'dpo_data_path', None)}  "
        f"dpo_prob={getattr(training_arguments, 'dpo_prob', 0.0)}",
        flush=True
    )

    logger_setting(getattr(training_arguments, "output_dir", None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)

    model_cfg = TinyLlavaConfig()
    model_cfg.load_from_config(model_arguments)
    model = _load_tinyllava_from_recipe(training_recipe, model_args, model_cfg, training_arguments.pretrained_model_path)
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    
    
    # tinyllava/train/train_dpo.py  【插入位置：load 完 model + recipe 后，取 tokenizer 前】
    # --- safety guard for tokenizer & embeddings ---
    
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        # 如你的模型类自带构造 tokenizer 的方法，可在这里补；否则报错更直观
        raise RuntimeError("model.tokenizer is None; please ensure recipe.load() sets tokenizer for flat checkpoints.")

    # pad_token 兜底
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Resize（重复调用也安全；若无新增 token 会返回 0 改动）
    try:
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tok))
        elif hasattr(model, "language_model"):
            model.language_model.resize_token_embeddings(len(tok))
    except Exception as e:
        print(f"[WARN] resize_token_embeddings failed: {e}", flush=True)

    # 最终传给数据模块使用
    tokenizer = tok


    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True

    # --- 生成包含 SFT+DPO 的数据模块（保持原接口）
    data_module = make_dpo_data_module(tokenizer=tokenizer, data_args=data_arguments)

    # ---- 取出 “纯 SFT 训练集” 与 “DPO 对数据集” ----
    
    train_ds_sft = data_module.get("train_dataset_sft", None)
    train_ds_dpo = data_module.get("train_dataset_dpo", None)

    if train_ds_sft is None or train_ds_dpo is None:
        mixed = data_module.get("train_dataset", None)
        if mixed is None:
            raise RuntimeError("make_dpo_data_module must return 'train_dataset' or explicit 'train_dataset_sft/train_dataset_dpo'.")

        sft_tmp, dpo_tmp = _split_sft_dpo_from_mixed(mixed)

        # --- 情况A：可按属性切分成功
        if sft_tmp is not None or dpo_tmp is not None:
            train_ds_sft = train_ds_sft or sft_tmp
            train_ds_dpo = train_ds_dpo or dpo_tmp

        # --- 情况B：无法按属性切分（没有 sft_len/dpo_len）
        else:
                # 退化为：把整个 mixed 当作 SFT；DPO 置空
            train_ds_sft = mixed
            train_ds_dpo = None

            # 只有“确实想用 DPO（给了 dpo_data_path）且 dpo_prob>0”才告警并置零
            want_dpo = bool(getattr(data_arguments, "dpo_data_path", None))
            dpo_prob = float(getattr(training_arguments, "dpo_prob", 0.0) or 0.0)

            if want_dpo and dpo_prob > 0.0:
                print("[WARN] DPO requested (has dpo_data_path & dpo_prob>0) but dataset doesn't expose sft_len/dpo_len; "
                      "falling back to PURE SFT and setting dpo_prob=0.", flush=True)
                setattr(training_arguments, "dpo_prob", 0.0)

        

    # 二次防御：SFT 仍为空则报错
    if train_ds_sft is None:
        raise RuntimeError("SFT training dataset is None. Please check your data module.")

    # ---- 参考模型（可选）
    ref_model = None
    if getattr(training_arguments, "use_ref_model", False):
        ref_cfg = TinyLlavaConfig()
        ref_cfg.load_from_config(model_arguments)
        saved_root = training_arguments.ref_model_root if getattr(training_arguments, "ref_model_root", None) else training_arguments.pretrained_model_path
        ref_model = _load_tinyllava_from_recipe(training_recipe, model_args, ref_cfg, saved_root)
        ref_model.config.use_cache = False
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # ---- DPO 对数据的 DataLoader（与 SFT 主链路完全独立）
    dpo_loader = None
    if train_ds_dpo is not None and getattr(training_arguments, "dpo_prob", 0.0) > 0.0:
        per_device = _int_from_env("DPO_PER_DEVICE_BATCH", getattr(training_arguments, "per_device_train_batch_size", 1))
        dpo_loader = DataLoader(
            train_ds_dpo,
            batch_size=per_device,
            shuffle=True,
            drop_last=True,
            num_workers=training_arguments.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=training_arguments.dataloader_num_workers > 0,
            collate_fn=DataCollatorForDPODialogue(tokenizer=tokenizer),  # ★ 加上这个
        )
    
    # ---- 统计参数量
    n_all = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] trainable {n_trn}/{n_all} ({(n_trn/n_all if n_all else 0):.2%})", flush=True)

    # ---- 梯度检查点（保持与 train.py 一致）
    if getattr(training_arguments, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            get_emb = getattr(model, "get_input_embeddings", None)
            if callable(get_emb):
                emb = get_emb()
                def _make_req_grad(module, inp, out):
                    if hasattr(out, "requires_grad") and not out.requires_grad:
                        out.requires_grad_(True)
                emb.register_forward_hook(_make_req_grad)

    # ---- 构造 Trainer：SFT 为主链路，DPO 按概率追加
    num_all   = sum(p.numel() for p in model.parameters())
    num_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] trainable {num_train}/{num_all} ({num_train/num_all:.2%})")


    # ---- 统计参数量
    n_all = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] trainable {n_trn}/{n_all} ({(n_trn/n_all if n_all else 0):.2%})", flush=True)

    # === 新增：为零就早停，避免 DeepSpeed 空 param_groups 触发 IndexError ===
    if n_trn == 0:
        first_trainables = [n for n, p in model.named_parameters() if p.requires_grad][:30]
        raise RuntimeError(
            "[SANITY] No trainable parameters after applying training recipe. "
            "If vision/connector are frozen, you must ensure LoRA is attached to LLM. "
            f"first_trainables={first_trainables}"
        )
        
    trainer = LLaVADPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_ds_sft,          # ★ SFT 主数据（与 train.py 路线一致）
        eval_dataset=data_module.get("eval_dataset", None),
        data_collator=data_module.get("data_collator", None),
        # DPO 附加项
        dpo_dataloader=dpo_loader,
        ref_model=ref_model,
        dpo_prob=getattr(training_arguments, "dpo_prob", 0.0),
        dpo_weight=getattr(training_arguments, "dpo_weight", 1.0),
        dpo_beta=getattr(training_arguments, "dpo_beta", 0.1),
        dpo_margin=getattr(training_arguments, "dpo_margin", 0.0),
        dpo_label_smoothing=getattr(training_arguments, "label_smoothing", 0.0),
        dpo_ipo=getattr(training_arguments, "ipo", False),
    )

    # opt = trainer.optimizer  # 或你手动构造的 optimizer 变量
    # print("[OPT] num_param_groups:", len(opt.param_groups))
    # print("[OPT] group0 size:", sum(p.numel() for p in opt.param_groups[0]["params"]))

    trainer.add_callback(ProgressCallback(getattr(training_arguments, "output_dir", None)))
    trainer.train()
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    try:
        train()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc(file=_ERR_FH)
        raise
    
    # train()
