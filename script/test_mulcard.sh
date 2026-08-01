export HCCL_BUFFSIZE=1024
export VLLM_VERSION="0.25.1"
# 更换 HCCL 的基础通信端口（避开已被占用的 60000）
export HCCL_IF_BASE_PORT=60050
export ASCEND_RT_VISIBLE_DEVICES="4,5,6,7"

export PYTHONPATH=/home/g00619970/moeoffload_multi/vllm:${PYTHONPATH}


# 适当延长多卡建链超时时间（单位：秒）
export HCCL_CONNECT_TIMEOUT=600
vllm serve /home/g00955623/weights/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 8150 \
  --seed 1024 \
  --max_model_len 150000 \
  --safetensors-load-strategy 'prefetch' \
  --max-num-batched-tokens 8192  \
  --served-model-name glm-5 \
  --gpu-memory-utilization 0.95 \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --no-async-scheduling \
  --max-num-seqs 1 \
  --block-size 128 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --api-server-count 1 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
  --additional_config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess":false}' \
  # 2>&1 | tee run_online.log


  # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  # --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \

    # --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/z00828031/code/dsv4_merge/profile", "torch_profiler_with_stack": false}' \