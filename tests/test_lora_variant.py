"""LoRA <-> klein variant guardrail (core.lora_flux2.lora_variant / check_lora_compatibility /
assert_lora_matches_model).

Regression: a klein-9B LoRA (hidden 4096, 8 double / 24 single blocks) was selectable with the
loaded klein-4B model (hidden 3072, 5 / 20) and failed at generate time with a raw state_dict
size-mismatch dump; check_lora_compatibility also rejected every 9B LoRA (single index >= 20).
"""
import pytest
import torch
from safetensors.torch import save_file


def _bfl(hidden, n_double, n_single, r=4):
    sd = {}
    for d in range(n_double):
        sd[f"diffusion_model.double_blocks.{d}.img_attn.qkv.lora_A.weight"] = torch.zeros(r, hidden)
        sd[f"diffusion_model.double_blocks.{d}.img_attn.qkv.lora_B.weight"] = torch.zeros(3 * hidden, r)
        sd[f"diffusion_model.double_blocks.{d}.img_attn.proj.lora_A.weight"] = torch.zeros(r, hidden)
        sd[f"diffusion_model.double_blocks.{d}.img_attn.proj.lora_B.weight"] = torch.zeros(hidden, r)
    for s in range(n_single):
        sd[f"diffusion_model.single_blocks.{s}.linear1.lora_A.weight"] = torch.zeros(r, hidden)
        sd[f"diffusion_model.single_blocks.{s}.linear1.lora_B.weight"] = torch.zeros(9 * hidden, r)
        sd[f"diffusion_model.single_blocks.{s}.linear2.lora_A.weight"] = torch.zeros(r, 4 * hidden)
        sd[f"diffusion_model.single_blocks.{s}.linear2.lora_B.weight"] = torch.zeros(hidden, r)
    return sd


def _diffusers(hidden, n_single, r=4):
    sd = {}
    for s in range(n_single):
        p = f"transformer.single_transformer_blocks.{s}.attn"
        sd[f"{p}.to_qkv_mlp_proj.lora_A.weight"] = torch.zeros(r, hidden)
        sd[f"{p}.to_qkv_mlp_proj.lora_B.weight"] = torch.zeros(9 * hidden, r)
        sd[f"{p}.to_out.lora_A.weight"] = torch.zeros(r, 4 * hidden)
        sd[f"{p}.to_out.lora_B.weight"] = torch.zeros(hidden, r)
    return sd


def _save(tmp_path, sd, name="l.safetensors"):
    p = tmp_path / name
    save_file(sd, str(p))
    return str(p)


def test_variant_bfl_4b_and_9b(tmp_path):
    from core.lora_flux2 import lora_variant
    assert lora_variant(_save(tmp_path, _bfl(3072, 5, 20))) == "4b"
    assert lora_variant(_save(tmp_path, _bfl(4096, 8, 24), "b.safetensors")) == "9b"


def test_variant_diffusers_format(tmp_path):
    from core.lora_flux2 import lora_variant
    assert lora_variant(_save(tmp_path, _diffusers(3072, 3))) == "4b"
    assert lora_variant(_save(tmp_path, _diffusers(4096, 3), "b.safetensors")) == "9b"


def test_variant_lora_down_up_naming(tmp_path):
    from core.lora_flux2 import lora_variant
    sd = {k.replace("lora_A", "lora_down").replace("lora_B", "lora_up"): v
          for k, v in _bfl(4096, 2, 2).items()}
    assert lora_variant(_save(tmp_path, sd)) == "9b"


def test_variant_unknown_for_other_hidden_size(tmp_path):
    from core.lora_flux2 import lora_variant
    assert lora_variant(_save(tmp_path, _bfl(6144, 2, 2))) is None


def test_compat_accepts_9b_block_counts(tmp_path):
    from core.lora_flux2 import check_lora_compatibility
    check_lora_compatibility(_save(tmp_path, _bfl(4096, 8, 24)))   # must not raise
    check_lora_compatibility(_save(tmp_path, _bfl(3072, 5, 20), "b.safetensors"))


def test_compat_rejects_flux2_dev_block_counts(tmp_path):
    from core.lora_flux2 import check_lora_compatibility
    with pytest.raises(RuntimeError, match="larger model"):
        check_lora_compatibility(_save(tmp_path, _bfl(6144, 8, 48)))


class _Cfg:
    def __init__(self, heads, dim):
        self.num_attention_heads, self.attention_head_dim = heads, dim


class _Pipe:
    def __init__(self, heads):
        self.transformer = type("T", (), {"config": _Cfg(heads, 128)})()


def test_model_variant_from_pipe():
    from core.lora_flux2 import pipe_variant
    assert pipe_variant(_Pipe(24)) == "4b"
    assert pipe_variant(_Pipe(32)) == "9b"


def test_assert_matches_model_raises_clear_error(tmp_path):
    from core.lora_flux2 import assert_lora_matches_model
    p9 = _save(tmp_path, _bfl(4096, 2, 2), "eyes_9b.safetensors")
    with pytest.raises(RuntimeError) as e:
        assert_lora_matches_model(_Pipe(24), p9)
    msg = str(e.value)
    assert "eyes_9b.safetensors" in msg and "9B" in msg and "4B" in msg
    assert_lora_matches_model(_Pipe(32), p9)   # matching: no raise


def test_assert_matches_model_lets_unknown_variant_through(tmp_path):
    from core.lora_flux2 import assert_lora_matches_model
    assert_lora_matches_model(_Pipe(24), _save(tmp_path, _bfl(6144, 1, 1)))
