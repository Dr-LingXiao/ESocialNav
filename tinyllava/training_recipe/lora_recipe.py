import os
from collections import OrderedDict

import torch
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.tuners.lora import LoraLayer

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import *
from ..utils import log
from ..model import TinyLlavaConfig, TinyLlavaForConditionalGeneration


@register_training_recipe('lora')
class LoRATrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments
        self.lora_skip_module = ['connector', 'vision_tower', 'language_model']

    def training_model_converse(self, model):
        if self.training_arguments.tune_type_connector == 'lora':
            self.lora_skip_module.remove('connector')
        if self.training_arguments.tune_type_llm == 'lora':
            self.lora_skip_module.remove('language_model')
        if self.training_arguments.tune_type_vision_tower == 'lora':
            self.lora_skip_module.remove('vision_tower')

        lora_config = LoraConfig(
            r=self.training_arguments.lora_r,
            lora_alpha=self.training_arguments.lora_alpha,
            target_modules=find_all_linear_names(model, self.lora_skip_module),
            lora_dropout=self.training_arguments.lora_dropout,
            bias=self.training_arguments.lora_bias,
            task_type="CAUSAL_LM",
        )

        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)

        # 如果模型已带有 LoRA（复用场景），不要再次挂载
        if not hasattr(model, 'peft_config') or model.peft_config is None:
            log("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
            
        # 🔴 关键：只让名字里包含 "lora" 的参数可训练，其它全部冻结
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 可选 debug
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        log(f"[LoRA] trainable param count = {len(trainable)}")
        log(f"[LoRA] first 10 trainable = {trainable[:10]}")

        return model
    
    def save(self, model, trainer):
        print(f"[SAVE-DBG] out={self.training_arguments.output_dir} "
              f"pretrained_model_path={self.training_arguments.pretrained_model_path} "
              f"exists={os.path.exists(self.training_arguments.pretrained_model_path) if self.training_arguments.pretrained_model_path else None}")

        model.config.use_cache = True
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        trainer.save_state()

        # finetune 阶段：是否 merge 由 save_merged_lora 控制（默认 False，便于后续复用）
        if 'finetune' in self.training_arguments.output_dir:
            if trainer.deepspeed:
                torch.cuda.synchronize()

            real_model = getattr(model, "module", model)
            merge_flag = getattr(self.training_arguments, "save_merged_lora", False)

            try:
                if isinstance(real_model, PeftModel) or getattr(real_model, "peft_config", None) is not None:
                    if merge_flag:
                        print("[SAVE] Merging LoRA adapters into base weights...")
                        real_model = real_model.merge_and_unload()
                        if hasattr(model, "module"):
                            model.module = real_model
                        else:
                            model = real_model
                    else:
                        print("[SAVE] Keeping LoRA adapters (no merge); saving base+adapter.")
            except Exception as e:
                print(f"[WARN] LoRA merge decision failed, continue saving: {e}")

            trainer.save_model(self.training_arguments.output_dir)
            return

        # pretrain 或非 finetune：分离保存（基座 + 适配器）
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.language_model.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(language_model_output_dir, 'pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)

        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(vision_tower_output_dir, 'pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)

        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.connector.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(connector_output_dir, 'pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)

        lora_state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), self.training_arguments.lora_bias)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
