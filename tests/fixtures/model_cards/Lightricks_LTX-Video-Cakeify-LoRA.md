# Cakeify Effect LoRA for LTX Video

This repository contains a LoRA model trained on top of [LTX Video v0.9.5](https://huggingface.co/Lightricks/LTX-Video) to create "cakeify"-style video effects.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
    <img src="assets/examples/3d-character.gif" width="256">
    <img src="assets/examples/cat.gif" width="256">
    <img src="assets/examples/iphone.gif" width="256">
    <img src="assets/examples/pikachu.gif" width="256">
</div>

## Usage

For best results, use prompts with the "CAKEIFY" trigger word in the following format:

```
CAKEIFY a person using a knife to cut a cake shaped like [your object]
```

### ComfyUI Workflow

A reference ComfyUI workflow is provided in [assets/ltxv-i2v-lora.json](assets/ltxv-i2v-lora.json). You can import this workflow directly into ComfyUI to get started.

![ComfyUI Workflow](assets/comfyui-workflow.png)

Example input images that work well with this model can be found in the [assets/images](https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA/tree/main/assets/images) directory.

## Base Model

This LoRA is designed to work with [LTX Video v0.9.5](https://huggingface.co/Lightricks/LTX-Video). Make sure you have the base model loaded in ComfyUI before using this LoRA.

## Training Dataset

The model was trained using the [Cakeify Dataset](https://huggingface.co/datasets/Lightricks/Cakeify-Dataset), which is publicly available on Hugging Face.

---

## License

Please refer to the license of the base LTX Video model for usage terms and conditions.
