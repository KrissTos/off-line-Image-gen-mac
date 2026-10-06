import pytest


def test_all_loaded():
    import app
    assert app.zimage_lora_status(["a.safetensors"], []) == "Loaded 1 LoRA(s): a.safetensors"


def test_partial_failure_is_not_a_success_status():
    import app
    msg = app.zimage_lora_status(["a.safetensors"], ["b.safetensors", "c.safetensors"])
    assert not msg.startswith("Loaded")
    assert "b.safetensors" in msg and "c.safetensors" in msg
    with pytest.raises(RuntimeError):
        app.ensure_loras_loaded(msg)


def test_nothing_loaded():
    import app
    assert app.zimage_lora_status([], ["b.safetensors"]).startswith("LoRA load failed")
