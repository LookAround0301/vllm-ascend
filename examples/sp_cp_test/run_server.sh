nic_name="enp48s3u1u1"
local_ip=141.61.39.141

export HCCL_IF_IP=$local_ip         # 指定HCCL通信库使用的网卡 IP 地址
export GLOO_SOCKET_IFNAME=$nic_name # 指定使用 Gloo通信库时指定网络接口名称 
export TP_SOCKET_IFNAME=$nic_name   # 指定 TensorParallel使用的网络接口名称
export HCCL_SOCKET_IFNAME=$nic_name # 指定 HCCL 通信库使用的网络接口名称
export OMP_PROC_BIND=false          # 允许操作系统调度线程在多个核心之间迁移
export OMP_NUM_THREADS=100          # 在支持 OpenMP 的程序中，最多使用 100 个 CPU 线程进行并行计算
export VLLM_USE_V1=1                # 强制使用v1模型加载/推理路径
export HCCL_BUFFSIZE=1024            # 每个通信操作的缓冲区大小为 1024 Bytes
# export VLLM_TORCH_PROFILER_DIR="/home/l00889328/profiling" # profiling保存路径
export ASCEND_LAUNCH_BLOCKING=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_VERSION="0.11.0"

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /mnt/nfs/weight/Qwen3-Coder-480B-A35B-Instruct-W8A8  \
  --host $local_ip \
  --port 8004 \
  --served-model-name qwen3-480b \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --prefill_context_parallel_size 2 \
  --decode_context_parallel_size 2 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --max-num-seqs 1 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enforce-eager \
  --quantization "ascend" \
#   --compilation-config '{"cudagraph_capture_sizes":[1,2,5,10,64],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"ascend_scheduler_config":{"enabled":false}}'