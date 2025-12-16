import os
import torch

from ..utils import *
from ..model import *

class BaseTrainingRecipe:

    def __init__(self, training_arguments):
        self.training_arguments = training_arguments

    def __call__(self, model):
        model = self.training_model_converse(model)
        model = self.tune_type_setting(model)
        model.config.tune_type_connector = self.training_arguments.tune_type_connector
        model.config.tune_type_vision_tower = self.training_arguments.tune_type_vision_tower
        model.config.tune_type_llm = self.training_arguments.tune_type_llm
        model.config.tune_vision_tower_from_layer = self.training_arguments.tune_vision_tower_from_layer
        return model

    def add_args(self, model_args):
        llm_dtype = (torch.float16 if self.training_arguments.fp16 else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32))
        model_args['llm'].update(dict(torch_dtype=llm_dtype))
        if self.training_arguments.pretrained_model_path is not None:
            model_args['llm'].update(dict(pretrained_llm_path=os.path.join(self.training_arguments.pretrained_model_path, 'language_model')))
            model_args['vision_tower'].update(dict(pretrained_vision_tower_path=os.path.join(self.training_arguments.pretrained_model_path, 'vision_tower')))
            model_args['connector'].update(dict(pretrained_connector_path=os.path.join(self.training_arguments.pretrained_model_path, 'connector')))
        return model_args

    def tune_type_setting(self, model):
        model = self._llm_tune_type_setting(model)
        model = self._vision_tower_tune_type_setting(model)
        model = self._connector_tune_type_setting(model)
        return model

    # def _llm_tune_type_setting(self, model):
    #     tune_type = self.training_arguments.tune_type_llm.lower()
    #     assert tune_type in ('frozen', 'full', 'lora', 'qlora'), \
    #         f'tune_type {tune_type} not supported in this training recipe!'

    #     if tune_type == 'full':
    #         # 全参数训练：LLM 全部可训练
    #         model.language_model.requires_grad_(True)
    #     elif tune_type == 'frozen':
    #         # 完全冻结 LLM
    #         model.language_model.requires_grad_(False)
    #     # 🔴 lora / qlora：这里不要再动 requires_grad
    #     #    让 LoRATrainingRecipe / QLoRATrainingRecipe 在 training_model_converse 里自己控制

    #     # 保留原来的 gradient checkpointing 支持
    #     self.support_gradient_checkpoint(model.language_model, self.training_arguments.gradient_checkpointing)
    #     return model

    def _llm_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_llm.lower()
        assert tune_type in ('frozen', 'full', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        if tune_type == 'full':
            model.language_model.requires_grad_(True)
        elif tune_type == 'frozen':
            model.language_model.requires_grad_(False)
        self.support_gradient_checkpoint(model.language_model, self.training_arguments.gradient_checkpointing)
        return model

    def _vision_tower_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_vision_tower.lower()
        assert tune_type in ('frozen', 'full', 'partially-tune', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        if tune_type == 'full':
            model.vision_tower.requires_grad_(True)
        elif tune_type == 'frozen':
            model.vision_tower.requires_grad_(False)
        elif tune_type == 'partially-tune':
            from_layer = self.training_arguments.tune_vision_tower_from_layer
            if from_layer > -1:
                log(f'Tune the vision tower from layer {from_layer}!')
                for n, p in model.vision_tower.named_parameters():
                    if 'vision_model.encoder.layers.' in n:
                        layer_id = int(n.split('vision_model.encoder.layers.')[-1].split('.')[0])
                        p.requires_grad = (layer_id >= from_layer)
                    else:
                        p.requires_grad = False
        return model

    def _connector_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_connector.lower()
        assert tune_type in ('frozen', 'full', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        if tune_type == 'full':
            for p in model.connector.parameters():
                p.requires_grad = True
        elif tune_type == 'frozen':
            for p in model.connector.parameters():
                p.requires_grad = False
        return model

    def training_model_converse(self, model):
        return model

    def save(self, model, trainer):
        model.config.use_cache = True
        # save tokenizer
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        # save entire model config
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        # save trainer
        trainer.save_state()

        # finetune 阶段：整包保存（是否 merge 由 lora_recipe 控制）
        if 'finetune' in self.training_arguments.output_dir and self.training_arguments.pretrained_model_path is not None:
            if trainer.deepspeed:
                torch.cuda.synchronize()
            trainer.save_model(self.training_arguments.output_dir)
            return
        
        # if 'finetune' in self.training_arguments.output_dir and self.training_arguments.pretrained_model_path is not None:
        #     if trainer.deepspeed:
        #         torch.cuda.synchronize()
        #     trainer.save_model(self.training_arguments.output_dir)
        #     return

        # pretrain 阶段分别保存各子模块（非 LoRA）
        language_model_state_dict = get_state_maybe_zero_3(model.language_model.named_parameters(), [''], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)

        vision_tower_state_dict = get_state_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), [''], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            if isinstance(model.vision_tower._vision_tower, PreTrainedModel):
                model.vision_tower._vision_tower.config.save_pretrained(vision_tower_output_dir, from_pt=True)

        connector_state_dict = get_state_maybe_zero_3(model.connector.named_parameters(), [''], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)

    def load(self, model, model_args={}):
        # 既支持根目录，也支持 lora_adapter/ 子目录
        ckpt_root = self.training_arguments.pretrained_model_path
        has_root_adapter = (ckpt_root is not None) and os.path.exists(os.path.join(ckpt_root, 'adapter_config.json'))
        has_sub_adapter  = (ckpt_root is not None) and os.path.exists(os.path.join(ckpt_root, 'lora_adapter', 'adapter_config.json'))
        is_lora_ckpt = has_root_adapter or has_sub_adapter

        if not is_lora_ckpt:
            # 非 LoRA：保持原逻辑，不影响非 lora pretrain
            model.load_llm(**model_args['llm'])
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
            return model

        # LoRA 情况：优先使用你保存下来的 base（language_model/）；没有就回退到 hub 模型
        base_llm_src = model_args['llm'].get('pretrained_llm_path') or model_args['llm']['model_name_or_path']
        model.language_model = model.language_model.from_pretrained(
            base_llm_src,
            attn_implementation='flash_attention_2',
            torch_dtype=model_args['llm']['torch_dtype']
        )
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])
        model.to(model_args['llm']['torch_dtype'])

        from peft import PeftModel
        adapter_dir = os.path.join(ckpt_root, 'lora_adapter') if has_sub_adapter else ckpt_root

        # 关键：第二阶段继续 LoRA 训练 → 不合并，直接以可训练方式挂上同一个 adapter
        train_with_lora = self.training_arguments.tune_type_llm.lower() in ('lora', 'qlora')
        print('Loading LoRA adapter from:', adapter_dir)
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=train_with_lora,
            adapter_name="default"
        )

        if train_with_lora:
            print('LoRA attached for training (no merge).')
        else:
            print('Merging LoRA weights for non-LoRA training/inference...')
            model = model.merge_and_unload()
            print('Merged.')

        print('Model is loaded...')
        return model

    # def load(self, model, model_args={}):
    #     # 当预训练路径不是 LoRA 形式时，按常规加载
    #     is_lora_ckpt = (
    #         self.training_arguments.pretrained_model_path is not None
    #         and os.path.exists(os.path.join(self.training_arguments.pretrained_model_path, 'adapter_config.json'))
    #     )
    #     if not is_lora_ckpt:
    #         model.load_llm(**model_args['llm'])
    #         model.load_vision_tower(**model_args['vision_tower'])
    #         model.load_connector(**model_args['connector'])
    #         return model

    #     # LoRA 权重：默认复用，不合并，便于继续训练
    #     model.language_model = model.language_model.from_pretrained(
    #         model_args['llm']['model_name_or_path'],
    #         attn_implementation='flash_attention_2',
    #         torch_dtype=model_args['llm']['torch_dtype']
    #     )
    #     model.load_vision_tower(**model_args['vision_tower'])
    #     model.load_connector(**model_args['connector'])
    #     model.to(model_args['llm']['torch_dtype'])

    #     from peft import PeftModel
    #     print('Loading LoRA weights...')
    #     model = PeftModel.from_pretrained(model, self.training_arguments.pretrained_model_path)

    #     reuse_lora = getattr(self.training_arguments, "reuse_lora", False)
    #     if reuse_lora:
    #         print('Keeping LoRA adapters attached (no merge) for reuse...')
    #     else:
    #         print('Merging LoRA weights (reuse_lora=False)...')
    #         model = model.merge_and_unload()
    #     print('Model is loaded...')
    #     return model

    def support_gradient_checkpoint(self, model, gradient_checkpointing=False):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        if gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
