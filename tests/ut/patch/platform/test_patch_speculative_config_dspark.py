from types import SimpleNamespace

import pytest
from transformers import Qwen3Config
from vllm.config import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.platform.patch_speculative_config import (
    _dspark_post_init,
    hf_config_override,
)


def test_legacy_qwen3_dspark_config_is_normalized_before_model_inspection():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
        pad_token_id=163839,
    )

    normalized = hf_config_override(config)

    assert SpeculativeConfig.hf_config_override is hf_config_override
    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert normalized.block_size == 7
    assert normalized.pad_token_id == 163839


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 8])
def test_qwen3_dspark_requires_exact_trained_block_size(
    monkeypatch: pytest.MonkeyPatch,
    num_speculative_tokens: int,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    draft_hf_config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        block_size=7,
        mask_token_id=163824,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=num_speculative_tokens,
    )

    with pytest.raises(ValueError, match=r"trained block_size \(7\)"):
        _dspark_post_init(config)


def test_qwen3_dspark_accepts_exact_trained_block_size(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    draft_hf_config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        block_size=7,
        mask_token_id=163824,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=7,
    )

    _dspark_post_init(config)

    assert draft_hf_config.ptd_token_id == 163824


def test_qwen3_dspark_rejects_non_kimi_block_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    draft_hf_config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        block_size=5,
        mask_token_id=163824,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=5,
    )

    with pytest.raises(ValueError, match=r"trained block_size=7"):
        _dspark_post_init(config)
