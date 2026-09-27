"""BFL-native FLUX.2 LoRA → diffusers key conversion (core.lora_flux2.convert_bfl_flux2_lora).

Regression: diffusers' own converter hardcodes FLUX.2-dev block counts (8 double / 48 single),
requires MLP keys and rejects embedder/modulation keys, so a klein-4B LoRA
(fal/flux-2-klein-4B-outpaint-lora: 5 double / 20 single, no MLP) raised KeyError and the app
generated WITHOUT the LoRA while still listing it in the result info.
"""
import torch


def _pair(sd, key, out_dim, in_dim=8, r=2):
    sd[f"base_model.model.{key}.lora_A.weight"] = torch.randn(r, in_dim)
    sd[f"base_model.model.{key}.lora_B.weight"] = torch.randn(out_dim, r)


def _klein_like(n_double=2, n_single=3):
    sd = {}
    for d in range(n_double):
        for a in ("img_attn", "txt_attn"):
            _pair(sd, f"double_blocks.{d}.{a}.qkv", 12)
            _pair(sd, f"double_blocks.{d}.{a}.proj", 4)
    for s in range(n_single):
        _pair(sd, f"single_blocks.{s}.linear1", 20)
        _pair(sd, f"single_blocks.{s}.linear2", 4)
    for k in ("img_in", "txt_in", "time_in.in_layer", "time_in.out_layer",
              "double_stream_modulation_img.lin", "double_stream_modulation_txt.lin",
              "single_stream_modulation.lin", "final_layer.linear"):
        _pair(sd, k, 4)
    return sd


def test_is_bfl_format():
    from core.lora_flux2 import is_bfl_flux2_lora
    assert is_bfl_flux2_lora(_klein_like())
    assert not is_bfl_flux2_lora({"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1)})


def test_block_counts_come_from_the_file():
    from core.lora_flux2 import convert_bfl_flux2_lora
    out = convert_bfl_flux2_lora(_klein_like(n_double=2, n_single=3))
    assert "transformer.single_transformer_blocks.2.attn.to_qkv_mlp_proj.lora_A.weight" in out
    assert not any(".single_transformer_blocks.3." in k for k in out)
    assert "transformer.transformer_blocks.1.attn.to_q.lora_B.weight" in out
    assert not any(".transformer_blocks.2." in k for k in out)


def test_fused_qkv_split_shares_A_and_chunks_B():
    from core.lora_flux2 import convert_bfl_flux2_lora
    sd = _klein_like(n_double=1, n_single=1)
    A = sd["base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight"]
    B = sd["base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight"]
    out = convert_bfl_flux2_lora(sd)
    p = "transformer.transformer_blocks.0.attn"
    for proj in ("to_q", "to_k", "to_v"):
        assert torch.equal(out[f"{p}.{proj}.lora_A.weight"], A)
    q, k, v = torch.chunk(B, 3, dim=0)
    assert torch.equal(out[f"{p}.to_q.lora_B.weight"], q)
    assert torch.equal(out[f"{p}.to_v.lora_B.weight"], v)
    assert f"{p}.add_k_proj.lora_B.weight" in out          # txt_attn → add_*_proj
    assert f"{p}.to_out.0.lora_A.weight" in out            # img_attn.proj
    assert f"{p}.to_add_out.lora_A.weight" in out          # txt_attn.proj


def test_embedder_modulation_and_final_layer_mapped():
    from core.lora_flux2 import convert_bfl_flux2_lora
    out = convert_bfl_flux2_lora(_klein_like(1, 1))
    for t in ("x_embedder", "context_embedder",
              "time_guidance_embed.timestep_embedder.linear_1",
              "time_guidance_embed.timestep_embedder.linear_2",
              "double_stream_modulation_img.linear", "double_stream_modulation_txt.linear",
              "single_stream_modulation.linear", "proj_out"):
        assert f"transformer.{t}.lora_A.weight" in out, t


def test_all_keys_consumed_and_unknown_key_raises():
    import pytest
    from core.lora_flux2 import convert_bfl_flux2_lora
    sd = _klein_like(1, 1)
    n_in = len(sd)
    out = convert_bfl_flux2_lora(sd)
    assert len(out) >= n_in                    # qkv fans out, nothing dropped
    sd = _klein_like(1, 1)
    _pair(sd, "mystery_layer", 4)
    with pytest.raises(ValueError, match="mystery_layer"):
        convert_bfl_flux2_lora(sd)


def test_mlp_keys_mapped_when_present():
    from core.lora_flux2 import convert_bfl_flux2_lora
    sd = _klein_like(1, 1)
    for m in ("img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"):
        _pair(sd, f"double_blocks.0.{m}", 4)
    out = convert_bfl_flux2_lora(sd)
    for m in ("ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out"):
        assert f"transformer.transformer_blocks.0.{m}.lora_A.weight" in out


def test_failed_lora_load_stops_generation():
    """app.load_loras reports failure as a status string; generate_image must not
    carry on without the LoRA (it used to, while still listing it in the result info)."""
    import pytest
    import app
    for bad in ("LoRA error: not compatible", "No LoRAs loaded",
                "LoRA load failed — check console for details",
                "LoRA requires FLUX.2-klein or Z-Image Full model"):
        with pytest.raises(RuntimeError, match="LoRA not applied"):
            app.ensure_loras_loaded(bad)
    app.ensure_loras_loaded("Loaded 1 LoRA(s): x.safetensors")   # no raise


def test_converter_error_surfaces_as_runtime_error(tmp_path):
    import pytest
    from safetensors.torch import save_file
    from core.lora_flux2 import load_loras
    sd = {k: v.contiguous() for k, v in _klein_like(1, 1).items()}
    sd["base_model.model.mystery_layer.lora_A.weight"] = torch.zeros(2, 8)
    p = tmp_path / "bad.safetensors"; save_file(sd, str(p))

    class _Pipe:
        def unload_lora_weights(self): pass
        def load_lora_weights(self, *a, **k): raise AssertionError("must not reach diffusers")
    with pytest.raises(RuntimeError, match="mystery_layer"):
        load_loras(_Pipe(), [{"path": str(p), "strength": 1.0}])


def test_removed_loras_are_unloaded(monkeypatch):
    """A request without LoRAs must unload adapters left from the previous request
    (generate_image only called load_loras when lora_files was non-empty)."""
    import app
    calls = []
    monkeypatch.setattr(app, "load_loras", lambda l, d: calls.append(list(l)) or "No LoRAs loaded")
    monkeypatch.setattr(app, "current_lora_paths", [{"path": "old.safetensors", "strength": 1.0}])
    app.sync_loras([], "mps")
    assert calls == [[]]

    calls.clear()
    monkeypatch.setattr(app, "current_lora_paths", [])
    app.sync_loras([], "mps")
    assert calls == []                      # nothing loaded, nothing to do

    monkeypatch.setattr(app, "load_loras", lambda l, d: calls.append(list(l)) or "Loaded 1 LoRA(s): a")
    app.sync_loras([{"path": "a", "strength": 1}], "mps")
    assert calls == [[{"path": "a", "strength": 1}]]
