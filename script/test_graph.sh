unset ftp_proxy
unset https_proxy
unset http_proxy

export VLLM_VERSION=0.26.0
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
export ASCEND_RT_VISIBLE_DEVICES="6,7"

export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_USE_FASTOKENS=1
# 同步的修改
# export ASCEND_LAUNCH_BLOCKING=1

nic_name="ens6f1np1"
local_ip="90.90.93.34"

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export PYTHONPATH=/home/g00619970/moeoffload_multi/vllm:/home/g00619970/moeoffload_multi/vllm-ascend:$PYTHONPATH

# MemFabric acc_offload 扩展库：offload.initialize() 运行时会从该路径 dlopen
# libmf_hybm_accoffload.so，缺失会导致 "offload launch load library failed"
# → RuntimeError: MemFabric LOCAL initialization failed (ret=-1)
# source /usr/local/memfabric_hybrid/set_env.sh

rm -rf ~/ascend/log/debug/plog/*
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000


# ============================================================================
# DeepSeek-V4-Flash (w4a8) · TP2/EP2 · expert offload（多卡）· 图模式 · pregate+LRU
# ----------------------------------------------------------------------------
# 与 test.sh 唯一区别：expert_prefetch_enabled true（pregate 开），即全栈配置。
# offload：multi_card, num_device_experts=32 (per_rank=16), cache_policy(LRU) 开。
# dies 14,15；port 9090；served-model-name deepseek。
#
# 显存备注（同 test.sh）：
#   84G 显存时 → 128 卡 --gpu-memory-utilization 0.62 / 112 卡 0.71 / 当前 0.9
# ============================================================================

# 在使用84G显存时，
# 如果当前是128，那 --gpu-memory-utilization 0.62 \
# 如果当前是112，那 --gpu-memory-utilization 0.71 \
vllm serve /home/g00955623/weights/DeepSeek-V4-Flash \
    --host 0.0.0.0 \
    --port 8150 \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "glm-5" \
    --max-num-seqs 1 \
    --max-model-len 270000 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --aggregate-engine-logging \
    --api-server-count 1 \
    --safetensors-load-strategy 'prefetch' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3]}' \
    --speculative-config '{"num_speculative_tokens": 2,"method": "mtp","enforce_eager": true}' \
    --additional-config '{"multistream_overlap_shared_expert": false,"enable_cpu_binding":true, "expert_offload_config": {"expert_offload": true, "enable_multi_card": false, "shard_per_rank": false,"num_device_experts": 84,"num_device_layers": 2, "cache_policy_enabled": true, "expert_prefetch_enabled": true, "expert_prefetch_num":1,"hot_expert_preload": false, "hot_experts_file": "expert_rank_gsm8k.json","expert_substitution_enabled": false,"expert_substitution_threshold": 0.25,"moe_offload_debug": true}}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/g00619970/moeoffload_multi/prefile", "torch_profiler_with_stack": false}' \
    # --quantization ascend \

    # --enforce_eager \
    # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    # --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \

    # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3]}' \
    # --speculative-config '{"num_speculative_tokens": 2,"method": "mtp","enforce_eager": true}' \

    # --additional-config '{"multistream_overlap_shared_expert": false,"enable_cpu_binding":true, "expert_offload_config": {"expert_offload": true,"num_device_experts": [84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84], "num_device_layers": 2, "cache_policy_enabled": true, "expert_prefetch_enabled": true, "expert_prefetch_num":1, "hot_expert_preload": true, "hot_experts_file": "expert_rank_gsm8k.json", "moe_offload_debug": true}}' \

    # --additional-config '{"multistream_overlap_shared_expert": false,"enable_cpu_binding":true, "expert_offload_config": {"expert_offload": true,"num_device_experts": [78, 78, 78, 120, 78, 78, 120, 78, 78, 78, 120, 78, 78, 78, 78, 78, 78, 78, 78, 120, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 120, 78, 78, 78, 120, 78, 78, 120, 256], "num_device_layers": 2, "cache_policy_enabled": true, "expert_prefetch_enabled": true, "expert_prefetch_num":1, "hot_expert_preload": true, "hot_experts_file": "expert_rank_gsm8k.json", "substitute_policy_enabled": true, "substitute_alpha": 0.1, "substitute_scoring_func": "sqrtsoftplus", "substitute_skip_layers": [0, 1, 2], "substitute_max_reuse": 0, "moe_offload_debug": true}}' \

    # --additional-config '{"multistream_overlap_shared_expert": false,"enable_cpu_binding":true, "expert_offload_config": {"expert_offload": true, "enable_multi_card": true, "num_device_experts": 48, "num_device_layers": 2, "cache_policy_enabled": true, "expert_prefetch_enabled": true, "expert_prefetch_num":1, "hot_expert_preload": true, "hot_experts_file": "expert_rank_gsm8k.json","moe_offload_debug": true}}' \

    # "expert_substitution_enabled": true,"expert_substitution_threshold": 0.02,

    # --additional-config '{"multistream_overlap_shared_expert": false,"enable_cpu_binding":true, "expert_offload_config": {"expert_offload": true, "enable_multi_card": true,"h2d_backend": "memfabric", "memfabric_pool_size_gib": 25,"memfabric_log_level": 1,"num_device_experts": 48,"num_device_layers": 2, "cache_policy_enabled": true, "expert_prefetch_enabled": true, "expert_prefetch_num":1,"hot_expert_preload": false, "hot_experts_file": "expert_rank_gsm8k.json","expert_substitution_enabled": false,"expert_substitution_threshold": 0.25,"moe_offload_debug": true}}' \
