# tinyllava/utils/arguments.py

from dataclasses import dataclass, field
from typing import Optional
import transformers  # <- 新增

# === 模型结构 & 微调策略 ===
@dataclass
class ModelArguments:
        cache_dir: Optional[str] = field(default=None)
    
        model_name_or_path: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer_name_or_path: Optional[str] = field(default=None)
        attn_implementation: Optional[str] = field(default=None)
        vision_tower: Optional[str] = field(default='')
        vision_tower2: Optional[str] = field(default='')
        connector_type: str = field(default='linear')
        
        mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
        mm_patch_merge_type: Optional[str] = field(default='flat')
        mm_vision_select_feature: Optional[str] = field(default="patch")
        resampler_hidden_size: Optional[int] = field(default=768)
        num_queries: Optional[int] = field(default=128)
        num_resampler_layers: Optional[int] = field(default=3)
        model_max_length: int = field(
            default=512,
            metadata={
                "help":
                    "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            },
        )
        tokenizer_use_fast: bool = field(default=False)
        tokenizer_padding_side: str = field(default='right')


 

# === 数据相关 ===
@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data. 支持逗号或竖线分隔多个 JSON。"}
    )

    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    conv_version: str = 'llava_v1'  # 或你在 SFT 里一直用的那个模板

    # 单路 SFT/DPO 原有
    # 混合训练新增
    sft_data_path: Optional[str] = None
    sft_ratio: float = 1.0  # 若你以后改成“每步都算两路”，这个仅作日志用途
    dpo_pair_mode: str = field(
        default="last",
        metadata={"help": "DPO 配对模式：last / all / random"}
    )

    # # dataloader
    # group_by_modality_length: bool = False
    # dataloader_num_workers: int = 0

    # 由模型带出的处理器
    image_processor: Optional[object] = None

# === 训练超参（HF 的 TrainingArguments 基础上扩展） ===
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default='common')
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    tune_vision_tower_from_layer: Optional[int] = field(default=10)
    tune_type_connector: str = field(default="full") # support only: frozen, full
    tune_embed_tokens: Optional[int] = field(default=False)
    
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = field(default=None)
    
    # 复用现有逻辑：用 recipe 从“多模块目录”加载（language_model/vision_tower/connector）

    # DPO / Mix 超参（脚本会传）
    sft_weight: float = 1.0
    dpo_beta: float = 0.2
    kto_beta: float = 0.2
    kl_weight: float = 0.0
    label_smoothing: float = 0.0
    use_ref_model: bool = False
    ref_model_root: Optional[str] = None
    dpo_per_device_train_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides per_device_train_batch_size for the DPO dataloader only."}
    )
 
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = field(default=None)
    # DPO 额外（兼容你现有 dpo trainer 用到的）
    dpo_margin: float = 0.3
    kto_margin: float = 0.3
    ipo: bool = False
    mix_batch_sft_ratio: float = field(default=None, metadata={"help": "每个 batch 中 SFT 样本比例（0~1）。"})
    
     # 仅 SFT 主链路，按概率附加 DPO 损失
    dpo_data_path: Optional[str] = field(default=None, metadata={"help": "Path to DPO pair dataset (optional)."})
    dpo_prob: float = field(default=0.0, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    dpo_weight: float = field(default=1.0, metadata={"help": "Weight for DPO loss when triggered."})
    dpo_beta: float = field(default=0.1, metadata={"help": "DPO/IPO temperature (beta)."})
    dpo_use_ref: bool = field(default=True, metadata={"help": "Use reference model in DPO (True) or no-ref variant (False)."})
    
    
    kto_data_path: Optional[str] = field(default=None, metadata={"help": "Path to DPO pair dataset (optional)."})
    kto_prob: float = field(default=0.0, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    kto_weight: float = field(default=1.0, metadata={"help": "Weight for DPO loss when triggered."})
    kto_beta: float = field(default=0.1, metadata={"help": "DPO/IPO temperature (beta)."})
    kto_use_ref: bool = field(default=True, metadata={"help": "Use reference model in DPO (True) or no-ref variant (False)."})
    
    
    ppo_clip_range: float = field(default=0.2, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    ppo_entropy_coef: float = field(default=0.01, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    ppo_kl_coef: float = field(default=0.02, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    ppo_adv_norm: bool = True
    ppo_sync_steps: int = 1
    ppo_max_new_tokens: int = 128
    ppo_temperature: float = field(default=0.7, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    ppo_top_p: float = field(default=0.9, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    ppo_alpha_sim: float = field(default=1.0, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})
    ppo_beta_sim: float = field(default=0.5, metadata={"help": "Per-step probability to add DPO loss (0.0 disables)."})

    # ===== GRPO generation & loss – canonical names =====
    grpo_num_generations: int = field(default=4, metadata={"help": "Group size (number of completions per prompt)."})
    grpo_scale_rewards: str = field(default="group", metadata={"help": "Reward scaling: group/none."})
    grpo_max_completion_length: int = field(default=128, metadata={"help": "Max new tokens per completion."})
    grpo_temperature: float = field(default=0.7, metadata={"help": "Sampling temperature for generation."})
    grpo_top_p: float = field(default=0.9, metadata={"help": "Top-p for nucleus sampling."})
    grpo_entropy_coef: float = field(default=0.02, metadata={"help": "Entropy bonus coefficient (a.k.a. beta)."})
    grpo_mask_truncated_completions: bool = field(default=True, metadata={"help": "Ignore loss on early-stopped tokens."})
    grpo_steps_per_generation: int = field(default=1, metadata={"help": "How often to (re)generate per training steps."})
    grpo_generation_batch_size: int = field(default=2, metadata={"help": "Batch size used during generation."})
    grpo_ds3_gather_for_generation: bool = field(default=True, metadata={"help": "If using DeepSpeed ZeRO-3, gather params for generation."})

    # ===== GRPO legacy-CLI compatibility (kept to match your current run script) =====
    # NOTE: 这些名字就是你命令里传的参数；保留它们可以避免 HfArgumentParser 报 unknown args。
    #       实际 Trainer 会优先读取上面的 grpo_*，其次读取这些旧名。
    loss_type: str = field(default="dapo", metadata={"help": "Kept for compatibility; not used by minimal GRPO trainer."})
    importance_sampling_level: str = field(default="sequence", metadata={"help": "Kept for compatibility."})
    clip_epsilon: float = field(default=0.2, metadata={"help": "Kept for compatibility (PPO-style)."})
    beta: float = field(default=0.02, metadata={"help": "Alias of grpo_entropy_coef for compatibility."})

    num_generations: int = field(default=4, metadata={"help": "Alias of grpo_num_generations."})
    scale_rewards: str = field(default="group", metadata={"help": "Alias of grpo_scale_rewards."})
    mask_truncated_completions: bool = field(default=True, metadata={"help": "Alias of grpo_mask_truncated_completions."})
    max_completion_length: int = field(default=128, metadata={"help": "Alias of grpo_max_completion_length."})
    temperature: float = field(default=0.7, metadata={"help": "Alias of grpo_temperature."})
    top_p: float = field(default=0.9, metadata={"help": "Alias of grpo_top_p."})
    steps_per_generation: int = field(default=1, metadata={"help": "Alias of grpo_steps_per_generation."})
    generation_batch_size: int = field(default=2, metadata={"help": "Alias of grpo_generation_batch_size."})
    ds3_gather_for_generation: bool = field(default=True, metadata={"help": "Alias of grpo_ds3_gather_for_generation."})

    
