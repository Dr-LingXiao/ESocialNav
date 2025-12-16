import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from .modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig
 
def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp
    

def load_pretrained_model(model_path: str,
                          torch_dtype: torch.dtype = torch.float16,
                          low_cpu_mem_usage: bool = True):
    """
    兼容两种权重形式：
    1) 模块化目录：language_model / vision_tower / connector 子目录分别保存
    2) 整包目录：一次性 save_pretrained() 出来的 TinyLLaVA（合并后的 MERGED 包）

    返回： (model, tokenizer, image_processor, context_len)
    """
    model_path = os.path.expanduser(model_path)

    # 先尝试把它当作“整包 TinyLLaVA”直接载入
    try:
        cfg = TinyLlavaConfig.from_pretrained(model_path)
        if getattr(cfg, "model_type", None) == "tinyllava":
            # 这是整包 TinyLLaVA
            model = TinyLlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            # 取出 image_processor（TinyLLaVA 内部在 load_vision_tower 时会挂到 _image_processor）
            image_processor = getattr(model.vision_tower, "_image_processor", None)
            if image_processor is None and hasattr(model.vision_tower, "get_image_processor"):
                image_processor = model.vision_tower.get_image_processor()
            # 上下文长度
            context_len = getattr(cfg, "tokenizer_model_max_length", None) or \
                          getattr(tokenizer, "model_max_length", 2048)
            return model, tokenizer, image_processor, context_len
    except Exception:
        # 如果不是整包，继续走模块化加载分支
        pass

    # 模块化分支：要求存在 language_model / vision_tower / connector 子目录
    lang_dir = os.path.join(model_path, "language_model")
    vt_dir   = os.path.join(model_path, "vision_tower")
    conn_dir = os.path.join(model_path, "connector")

    if not (os.path.isdir(lang_dir) and os.path.isdir(vt_dir) and os.path.isdir(conn_dir)):
        raise ValueError(
            f"'{model_path}' 既不是整包 TinyLLaVA（缺少 tinyllava 的 config），也不包含模块化子目录 "
            f"(language_model / vision_tower / connector)。请检查合并/导出是否完整。"
        )

    # 如果是模块化保存，按原有方式构建
    cfg = TinyLlavaConfig.from_pretrained(model_path)
    model = TinyLlavaForConditionalGeneration(cfg)
    # 语言模型/视觉塔/投影器从各自目录加载
    model.load_llm(pretrained_llm_path=lang_dir, torch_dtype=torch_dtype)
    model.load_vision_tower(pretrained_vision_tower_path=vt_dir)
    model.load_connector(pretrained_connector_path=conn_dir)

    # tokenizer：优先从根目录取（你合并脚本已经保存了）
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    image_processor = getattr(model.vision_tower, "_image_processor", None)
    if image_processor is None and hasattr(model.vision_tower, "get_image_processor"):
        image_processor = model.vision_tower.get_image_processor()

    context_len = getattr(cfg, "tokenizer_model_max_length", None) or \
                  getattr(tokenizer, "model_max_length", 2048)

    return model, tokenizer, image_processor, context_len


# def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
#                           device="cuda", **kwargs):
#     kwargs = {"device_map": device_map, **kwargs}
#     if device != "cuda":
#         kwargs['device_map'] = {"": device}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16
#     if model_name_or_path is not None and 'lora' not in model_name_or_path:
#         model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path,low_cpu_mem_usage=True)
        
#     elif model_name_or_path is not None and 'lora' in model_name_or_path:
#         if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
#             model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
#             model = TinyLlavaForConditionalGeneration(model_config)
#             language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
#             language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
#             model.language_model.load_state_dict(language_model_ckp)
#             vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
#             vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
#             model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
#             connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
#             connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
#             model.connector.load_state_dict(connector_ckp, strict=False)
#             model.to(torch.float16)
#             from peft import PeftModel
#             print('Loading LoRA weights...')
#             model = PeftModel.from_pretrained(model, model_name_or_path)
#             print('Merging LoRA weights...')
#             model = model.merge_and_unload()
#             print('Model is loaded...')
        
#     image_processor = model.vision_tower._image_processor
#     context_len = getattr(model.config, 'max_sequence_length', 2048)
#     # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
#     tokenizer = model.tokenizer
#     #tokenizer.pad_token = tokenizer.eos_token
#     return model, tokenizer, image_processor, context_len