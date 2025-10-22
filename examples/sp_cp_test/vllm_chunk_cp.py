import os
import time
import argparse
import random
import string
from pathlib import Path
import torch
 
from vllm import LLM, SamplingParams
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer


# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="8,9,10,11,12,13,14,15"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL"] = "1"
# os.environ["VLLM_LOGGING_LEVEL"]="DEBUG"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
os.environ["VLLM_VERSION"] = "0.11.0"


 
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--input_len', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=1024)
    parser.add_argument('--bs', type=int, default=1)
    # parser.add_argument('--model_path', type=str, default="/mnt/share/weights/DeepSeek-V2-Lite")
    parser.add_argument('--model_path', type=str, default="/mnt/nfs/weight/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument('--tp', type=int, default=8)   # 4 to 8
    parser.add_argument('--pcp', type=int, default=2)   # 4 to 8
    parser.add_argument('--dcp', type=int, default=2)   # 4 to 8
    parser.add_argument('--profiler_dir', type=str, default=None)
    parser.add_argument('-p', '--profiling', action="store_true")
    parser.add_argument('--iter_times', type=int, default=1)
    parser.add_argument('-c', '--enable_chunked_prefill', default=True)
 
    args = parser.parse_args()

    def generate_odd_queue_string(length):
        return ' '.join(str(2*i + 1) for i in range(length))

    
    def check_token_len(model_path, prompts):
        for i, prompt in enumerate(prompts):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>yyyyyyy")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            encoded = tokenizer(
                prompt,
                truncation=False,  # 不截断，我们自己控制长度
                return_tensors="pt"
            )
            
            # 获取当前样本的token长度
            sample_length = encoded["input_ids"].shape[1]
            print(f"提示词{i}的长度: {sample_length} tokens")
    
 
    # sampling_params = SamplingParams(temperature = 0.0, max_tokens=args.output_len)
    # sampling_params = SamplingParams(temperature = 0.8, top_p = 0.95, max_tokens=args.output_len)
    sampling_params = SamplingParams(temperature = 0.7, top_k = 20, top_p = 0.8, ignore_eos=True,  max_tokens=args.output_len)
    llm = LLM(model=args.model_path,
          trust_remote_code=True,
          enforce_eager=True,
          tensor_parallel_size=args.tp,  # tp=8
          prefill_context_parallel_size=args.pcp,
          decode_context_parallel_size=args.dcp,
          enable_prefix_caching=False,
          enable_expert_parallel=True,
          enable_chunked_prefill=True,
          max_num_batched_tokens=64, #1024, #16384  1024  74000  131072
          max_model_len=4096,   # 128K  131072
          additional_config={"ascend_scheduler_config": {"enabled": False}},
          max_num_seqs=1,
          block_size=128,
          gpu_memory_utilization=0.9  # Ä¬ÈÏÖµ0.9
          )
 
    base = 400
    for i in range(1):
        # prompts = [
        #     generate_odd_queue_string(base)+" " 
        # ]
        prompts = [
            "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            # "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", # (9 + 3) / 2
            # "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?" # (7 + 1) / 2 = 4
            # "While Joanne is gathering apples from her family\u2019s orchard, her sister comes outside to help her. Joanne gathers 30 apples from the tallest trees, half this amount from the shortest trees, and more apples from the average trees. Compared with Joanne, her sister gathers twice as many apples from the tallest trees and 3 times as many apples from the shortest trees. She doesn't take any from the average trees. If the sisters have gathered a combined total of 500 apples, how many apples did Joanne gather from the average trees? Please give me the final answer directly." # 4 / 2
        ]

        for i, prompt in enumerate(prompts):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>yyyyyyy")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            encoded = tokenizer(
                prompt,
                truncation=False,  # 不截断，我们自己控制长度
                return_tensors="pt"
            )
            
            # 获取当前样本的token长度
            sample_length = encoded["input_ids"].shape[1]
            print(f"提示词{i}的长度: {sample_length} tokens")


        t0 = time.time()
        for _ in range(args.iter_times):
            outputs = llm.generate(prompts, sampling_params)
        t1 = time.time()
        print(f"TTFT: {(t1 - t0) * 1000 / (args.iter_times * args.bs)} ms")
     
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(
            #     f"req_num: {i}\nGenerated text: {generated_text!r}"
            # )
            prompt = prompt.split(" ")
            print(
                #f"prompt:{prompt}\n"
                #f"req_num: {i}\n[{prompt}] -> Generated text: {generated_text!r}\n"
                f"req_num: {i}\n[{prompt[-5:]}] -> Generated text: {generated_text!r}\n"
                # f"Token ids: {output.outputs[0].token_ids}\n"
            )
     
    print("end.")
