import argparse
import time

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import re
import time
import statistics

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def to_gb(x_bytes):
    return x_bytes / (1024 ** 3)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor(image)
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        #print(prompt)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # >>> 只保留 Q4（question_id 以 _q4 结尾，或文案等于 "What should the robot do?"）
    def _is_q4(x):
        qid = x.get("question_id", "")
        qtxt = (x.get("text") or x.get("prompt") or x.get("question") or "").strip()
        return qid.endswith("_q4") or qtxt == "What should the robot do?"
    if getattr(args, "only_q4", False):
        questions = [x for x in questions if _is_q4(x)]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)
    model.to(device='cuda')
    
    model.eval()

    # ====== PARAMS / MEM (merged full model) ======
    total_p, trainable_p = count_params(model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # =============================================

    print(f"[MODEL] {args.model_path}")
    print(f"[PARAMS][INFER_TOTAL] {total_p/1e9:.3f}B  ({total_p/1e6:.3f}M)")
    print(f"[PARAMS][TRAINABLE]   {trainable_p/1e9:.3f}B  ({trainable_p/1e6:.3f}M)")


    # >>> 计时器（E2E 与 model-only）
    e2e_start = time.perf_counter()
    gen_time = 0.0
    n_done = 0

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            # >>> model-only 计时（需同步防止异步偏差）
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                image_sizes=image_sizes,
                use_cache=True)
            torch.cuda.synchronize()
            gen_time += (time.perf_counter() - t0)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_base,
                                   "metadata": {}}) + "\n")
        n_done += 1

    ans_file.close()

    # >>> 打印 Q4 的 FPS（只在有 --print-fps 时输出）
    if getattr(args, "print_fps", False) and n_done > 0:
        e2e_total = time.perf_counter() - e2e_start
    e2e_fps = n_done / e2e_total

    model_fps = n_done / gen_time if gen_time > 0 else float("nan")
    model_latency_ms = (gen_time / n_done) * 1000.0 if n_done > 0 else float("nan")

    peak_alloc_gb = to_gb(torch.cuda.max_memory_allocated())
    peak_resv_gb = to_gb(torch.cuda.max_memory_reserved())

    print(f"[Q4-BENCH] N={n_done}")
    print(f"[Q4-BENCH] E2E:        {e2e_fps:.3f} img/s   (total {e2e_total:.3f}s)")
    print(f"[Q4-BENCH] Model-only:  {model_fps:.3f} img/s   (gen   {gen_time:.3f}s)")
    print(f"[Q4-BENCH] Latency:     {model_latency_ms:.1f} ms/step")
    print(f"[Q4-BENCH] PeakMem:     alloc={peak_alloc_gb:.2f} GB, reserved={peak_resv_gb:.2f} GB")

    # 可选：写文件（跟 answers 放一起）
    try:
        bench_path = os.path.join(os.path.dirname(answers_file), "bench.log")
        with open(bench_path, "w") as f:
            f.write(f"MODEL={args.model_path}\n")
            f.write(f"PARAMS_INFER_TOTAL={total_p}\n")
            f.write(f"E2E_FPS={e2e_fps}\n")
            f.write(f"MODEL_FPS={model_fps}\n")
            f.write(f"LATENCY_MS_PER_STEP={model_latency_ms}\n")
            f.write(f"PEAK_ALLOC_GB={peak_alloc_gb}\n")
            f.write(f"PEAK_RESERVED_GB={peak_resv_gb}\n")
        print(f"[Q4-BENCH] Saved: {bench_path}")
    except Exception as e:
        print(f"[WARN] bench.log save failed: {e}")


    # if getattr(args, "print_fps", False) and n_done > 0:
    #     e2e_total = time.perf_counter() - e2e_start
    #     e2e_fps = n_done / e2e_total
    #     model_fps = n_done / gen_time if gen_time > 0 else float("nan")
    #     print(f"[Q4-FPS] N={n_done}  E2E: {e2e_fps:.3f} img/s (total {e2e_total:.3f}s)  "
    #           f"Model-only: {model_fps:.3f} img/s (gen {gen_time:.3f}s)")
        
# def eval_model(args):
#     # Model
#     disable_torch_init()
#     model_path = os.path.expanduser(args.model_path)
#     model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
#     text_processor = TextPreprocess(tokenizer, args.conv_mode)
#     data_args = model.config
#     image_processor = ImagePreprocess(image_processor, data_args)

#     questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
#     questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
#     answers_file = os.path.expanduser(args.answers_file)
#     os.makedirs(os.path.dirname(answers_file), exist_ok=True)
#     ans_file = open(answers_file, "w")


#     data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)
#     # print("Tokenizer's eos token: ", tokenizer.eos_token)
#     model.to(device='cuda')
#     for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
#         idx = line["question_id"]
#         cur_prompt = line["text"]
#         # keywords = [tokenizer.eos_token]
#         # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#         input_ids = input_ids.to(device='cuda', non_blocking=True)
#         with torch.inference_mode():
#             output_ids = model.generate(
#                 input_ids,
#                 images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
#                 pad_token_id=tokenizer.pad_token_id,
#                 do_sample=True if args.temperature > 0 else False,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 num_beams=args.num_beams,
#                 max_new_tokens=args.max_new_tokens,
#                 # stopping_criteria=[stopping_criteria],
#                 image_sizes=image_sizes,
#                 use_cache=True)

#         outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         # print("Printing outputs")
#         # print(outputs)
#         # time.sleep(5)
#         ans_id = shortuuid.uuid()
#         ans_file.write(json.dumps({"question_id": idx,
#                                    "prompt": cur_prompt,
#                                    "text": outputs,
#                                    "answer_id": ans_id,
#                                    "model_id": args.model_base,
#                                    "metadata": {}}) + "\n")
#         # ans_file.flush()
#     ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    # >>> 新增两个开关
    parser.add_argument("--only-q4", action="store_true",
                        help="Only run entries whose question_id endswith _q4 or prompt=='What should the robot do?'.")
    parser.add_argument("--print-fps", action="store_true",
                        help="Print Q4-only FPS (E2E and model-only).")
    
    args = parser.parse_args()

    eval_model(args)