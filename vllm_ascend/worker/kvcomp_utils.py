from collections import defaultdict
from vllm.model_executor.models.utils import extract_layer_index
import torch
from vllm.attention.layer import Attention


# largely follow vllm.v1.worker.utils.bind_kv_cache
def bind_hashk_cache(
    hashk_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_hashk_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated hashk cache to both ModelRunner and forward context so
    that the hashk cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_hashk_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in hashk_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        layer_name = layer_names[0]
        runner_hashk_caches.append(hashk_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, hashk_cache in hashk_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].hashk_cache = [hashk_cache]
