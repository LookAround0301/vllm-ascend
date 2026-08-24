#!/bin/bash

# ================================
# 清除代理
# ================================

unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY


# ================================
# 日志: stdout + stderr 全部落盘 (NPU/HCCL/aclnn/GE 的 fatal 都走 stderr)
# 直接 `bash test.sh` 或 `nohup bash test.sh &` 即可, 不要再在外层加重定向
# ================================
LOG_DIR="/home/g00619970/moeoffload_multi/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/model_$(date +%Y_%m%d_%H%M)_multimech.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[test.sh] logging stdout+stderr to: $LOG_FILE"


# ================================
# 网络配置
# ================================

local_ip="141.61.73.103"
node0_ip="141.61.73.103"
nic_name="eth0"

SERVICE_PORT=8150
DP_RPC_PORT=13389

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export PYTHONPATH=/home/g00619970/moeoffload_multi/vllm:/home/g00619970/moeoffload_multi/vllm-ascend:$PYTHONPATH

source /usr/local/Ascend/cann/set_env.sh

# ================================
# CANN 环境
# ================================

# source /home/w00887678/new_model_k3/CANN/cann-9.1.0/set_env.sh


# ================================
# 通信环境变量
# ================================

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
# 单步 execute_model 超时: 原值 30000s(~8h) 等于关掉了 worker 看门狗,
# worker 卡死时不报错、表现为静默 hang。调小到 600s, 卡死时 vLLM 会主动
# 报出是哪个 worker/step 超时 (单步正常是毫秒~秒级, 600s 足够)。
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600

# Kimi-K3 很大，主节点可以适当增加等待时间
export VLLM_ENGINE_READY_TIMEOUT_S=7200

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=800

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

# 卡死诊断: 打开 faulthandler + 允许 core dump
#  - 卡住时可 `kill -SIGQUIT <worker_pid>` 或 `py-spy dump --pid <pid>` 打印线程栈,
#    能直接看到卡在哪个 HCCL/aclnn 调用; 真崩溃时自动落 core 便于事后分析。
export PYTHONFAULTHANDLER=1
ulimit -c unlimited

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64

export HCCL_OP_EXPANSION_MODE="CCU_SCHED"

export VLLM_ASCEND_ENABLE_MLAPO=0
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# export ASCEND_LAUNCH_BLOCKING=1

# ================================
# 当前节点使用 8 张卡
# ================================

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# ================================
# DSpark
# ================================

SPECULATIVE_CONFIG="$(
printf \
'{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":8,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
"/mnt/share/y00823936/Inferact-Kimi-K3-DSpark"
)"


# ================================
# 启动 DP rank 0
# ================================
# /mnt/share/kimik3_0726/Kimi-K3
# /mnt/share/kimik3_0726/Kimi-K3-w4a8-mxfp-flex-quarot-0729
vllm serve /mnt/share/kimik3_0726/Kimi-K3 \
    --host 0.0.0.0 \
    --port ${SERVICE_PORT} \
    --served-model-name kimi \
    --allowed-local-media-path / \
    --trust-remote-code \
    --safetensors-load-strategy prefetch \
    --enable-expert-parallel \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address ${node0_ip} \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --tensor-parallel-size 8 \
    --max-num-seqs 1 \
    --max-model-len 1000000 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --tokenizer-mode kimi_k3 \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"expert_offload_config": {"expert_offload": true, "enable_multi_card": true, "num_device_experts": 272, "num_device_layers": 1, "cache_policy_enabled": true, "expert_prefetch_enabled": false, "expert_prefetch_num":1,"hot_expert_preload": false, "hot_experts_file": "expert_rank_gsm8k.json","expert_substitution_enabled": false,"expert_substitution_threshold": 0.25,"moe_offload_debug": false}}' \
    # --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/g00619970/moeoffload_multi/prefile","torch_profiler_with_stack":true}' \
    # --speculative-config "$SPECULATIVE_CONFIG"

    # "num_device_experts": 192 --max-model-len 133120 \     情况下
    # "num_device_experts": 240 --max-model-len 1080000\     情况下
        # NOTE: 投机解码(dspark)暂时禁用 —— draft 模型 Inferact-Kimi-K3-DSPark 的架构
    #       k3_dspark / K3DSparkModel 在 transformers 5.14.1 中未注册, 且模型目录缺少
    #       modeling_k3_dspark.py(trust_remote_code 也救不了), 启用会导致 SpeculativeConfig
    #       校验失败、进程启动即崩溃。拿到 draft 模型代码并注册后, 再放开下面这行。
# --enforce-eager \
    # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \


    

#     怎么用 / 怎么抓现场

#   启动方式改为(脚本自己会落 log,外层不要再加重定向):
#   nohup bash /home/g00619970/moeoffload_multi/script/test.sh &
#   或直接前台 bash test.sh。

#   下次卡死时,别急着杀进程,先抓栈(这是定位"卡在哪个调用"最直接的办法):
#   # 找到 worker 进程 pid
#   pgrep -af "VLLM::Worker"

#   # 二选一, 打印每个 worker 的线程栈
#   py-spy dump --pid <worker_pid>
#   # 或
#   kill -SIGQUIT <worker_pid>      # PYTHONFAULTHANDLER 会把所有线程栈打到 log 里
#   栈里会显示卡在哪个函数(典型是某个 hccl::... allreduce/all2all,或 aclnn... 内核,或 offload 的 H2D 拷贝),基本就能定位到死锁点。
