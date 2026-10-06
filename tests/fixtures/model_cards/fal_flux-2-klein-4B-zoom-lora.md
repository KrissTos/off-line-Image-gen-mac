
<Gallery />

<div align="center">

# 🔍 Flux Red Zoom LoRA

### Zoom and enhance red-highlighted regions of images

[![Try on fal.ai](https://img.shields.io/badge/🚀_Try_it_Live-fal.ai-FF6B35?style=for-the-badge)](https://fal.ai/models/fal-ai/flux-2/klein/4b/base/edit/lora)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

<img src="images/banner.jpg" alt="Red Zoom LoRA Banner" width="100%">

</div>

---

## 🎯 What does this model do?

This LoRA zooms into **red-highlighted areas** of images and generates an enlarged, detailed view. Simply mark the region you want to focus on with red, and the model will zoom and enhance it.
```
┌─────────────────────────────────────┐
│                                     │
│     ┌───────┐                       │
│     │ 🔴RED │  ──────────────►      │  Zoomed &
│     │  AREA │                       │  Enhanced
│     └───────┘                       │
│                                     │
└─────────────────────────────────────┘
```

Perfect for **detail extraction**, **region focus**, and **image enhancement**.

---

## 🖼️ Examples

<table>
<tr>
<th>Input (with red highlight)</th>
<th>Output (zoomed region)</th>
</tr>
<tr>
<td><img src="examples/example1_input.jpg" width="300"></td>
<td><img src="examples/example1_output.jpg" width="300"></td>
</tr>
<tr>
<td><img src="examples/example2_input.jpg" width="300"></td>
<td><img src="examples/example2_output.jpg" width="300"></td>
</tr>
<tr>
<td><img src="examples/example3_input.jpg" width="300"></td>
<td><img src="examples/example3_output.jpg" width="300"></td>
</tr>
<tr>
<td><img src="examples/example4_input.jpg" width="300"></td>
<td><img src="examples/example4_output.jpg" width="300"></td>
</tr>
</table>

---

## 🚀 Quick Start

### Prompt
```
Zoom into the red highlighted area
```

### Red Highlight Requirements
- Use red color to mark the area of interest
- The highlight should clearly indicate the zoom region
- Works best with rectangular or clearly defined areas

---

## 💻 Usage

### Try it Live on fal.ai

▶️ **[Open Playground](https://fal.ai/models/fal-ai/flux-2-klein/4b/base/edit/lora)**

### With fal.ai SDK
```python
import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

result = fal_client.subscribe(
    "fal-ai/flux-2/klein/4b/base/edit/lora",
    arguments={
        "prompt": "Zoom into the red highlighted area",
        "model_name": None,
        "loras": [{
            "path": "https://huggingface.co/fal/flux-2-klein-4B-zoom-lora/resolve/main/flux-red-zoom-lora.safetensors",
            "scale": 1.1
        }],
        "embeddings": [],
        "image_urls": ["https://your-image-with-red-highlight.png"]
    },
    with_logs=True,
    on_queue_update=on_queue_update,
)
print(result)
```

---

## 📦 Model Files

| File | Use Case |
|------|----------|
| `flux-red-zoom-lora.safetensors` | fal.ai |
| `aoqt0fLF25ymtU3S6pxrB_pytorch_lora_weights_comfy_converted.safetensors` | ComfyUI |

---

## 🎓 Training Details

<details>
<summary>Click to expand</summary>

### Dataset
- **Size**: 86 image pairs
- **Content**: Diverse images with zoom regions including:
  - **Landscapes**: nature details, architectural elements
  - **Objects**: product details, textures
  - **People**: facial features, accessories
  - **Art**: painting details, patterns
- **Red highlight**: Consistent red markers for zoom regions
- **Aspect ratios**: Various (1:1, 16:9, 9:16, 4:3, 3:4, etc.)

### Training
- **Base Model**: FLUX.2-Klein 4B
- **Platform**: [fal.ai](https://fal.ai)
- **Method**: LoRA training
- **Steps**: 4000
- **Learning Rate**: 0.00005

</details>

---

## 🎮 Use Cases

- **Detail Extraction**: Focus on specific parts of complex images
- **Product Showcase**: Highlight product features and details
- **Document Analysis**: Zoom into specific text or diagram regions
- **Creative Focus**: Create dramatic crop and zoom effects

---

## ⚠️ Limitations

- Works best with clearly defined red highlight areas
- Very small highlighted regions may lose detail
- Red objects in the image may be confused with highlight markers

---

## 📄 License

[Apache 2.0](https://opensource.org/licenses/Apache-2.0)

---

<div align="center">

### Created by [@ilkerzgi](https://huggingface.co/ilkerzgi) • Trained with [fal.ai](https://fal.ai)


[![Try on fal.ai](https://img.shields.io/badge/🚀_Try_it_Live-fal.ai-FF6B35?style=for-the-badge)](https://fal.ai/models/fal-ai/flux-2/klein/4b/base/edit/lora)

</div>
