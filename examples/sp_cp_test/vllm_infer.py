import argparse
import os
import time

from vllm import LLM, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL"] = "1"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
os.environ["VLLM_VERSION"] = "0.11.0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_len', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=2048)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="/mnt/nfs/weight/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--pcp', type=int, default=2)
    parser.add_argument('--dcp', type=int, default=2)
    # parser.add_argument('--tp', type=int, default=2)
    # parser.add_argument('--cp', type=int, default=1)
    # parser.add_argument('--dcp', type=int, default=2)
    parser.add_argument('--iter_times', type=int, default=1)

    args = parser.parse_args()

    # prompts = [
    #     "The capital of France is",
    #     "Hello, my name is Tom, I am",
    #     "The president of United States is",
    #     "AI future is"
    # ]


    prompts = [
            "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            # "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", # (9 + 3) / 2
            # "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?" # (7 + 1) / 2 = 4
            "While Joanne is gathering apples from her family\u2019s orchard, her sister comes outside to help her. Joanne gathers 30 apples from the tallest trees, half this amount from the shortest trees, and more apples from the average trees. Compared with Joanne, her sister gathers twice as many apples from the tallest trees and 3 times as many apples from the shortest trees. She doesn't take any from the average trees. If the sisters have gathered a combined total of 500 apples, how many apples did Joanne gather from the average trees?" # 4 / 2
        ]
    # sampling_params = SamplingParams(temperature = 0.0, max_tokens=args.output_len)
    sampling_params = SamplingParams(temperature = 0.8, top_p = 0.95, max_tokens=args.output_len)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=args.dcp,
        enable_prefix_caching=False,
        enable_expert_parallel=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=64,
        max_model_len=2048,
        # quantization="ascend",
        additional_config={"ascend_scheduler_config": {"enabled": False}},
        max_num_seqs=1,
        block_size=128,
        gpu_memory_utilization=0.9
    )

    t0 = time.time()
    for _ in range(args.iter_times):
        outputs = llm.generate(prompts, sampling_params)
    t1 = time.time()
    print(f"TTFT: {(t1 - t0) * 1000 / (args.iter_times * args.bs)} ms")

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"req_num: {i}\nGenerated text: {generated_text!r}")