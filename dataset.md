##  Overview on datasets 

**SmolVLM2-256M-Video** is a compact multimodal model capable of analyzing **video**, **image**, and **text** inputs to generate high-quality text outputs.  
It supports tasks like:

- Visual Question Answering  
- Video understanding  
- Multi-image reasoning  
- OCR-style Q&A  
- Story generation from visual input  

The model requires only **1.38 GB GPU RAM** for video inference — perfect for **edge devices**, **mobile**, and **low-compute** environments.

---

## 🧠 Uses

SmolVLM2 supports inference over **images**, **videos**, and **text prompts**.  
It allows interleaving media and text, enabling:

- Image captioning  
- Video description  
- VQA over images & videos  
- Multi-image comparison  
- Sequential visual storytelling  

⚠️ **Note:** The model does *not* generate images or videos.

---

## 📊 Evaluation Benchmarks

| Model Size | Video-MME | MLVU | MVBench |
|------------|-----------|------|---------|
| **2.2B** | 52.1 | 55.2 | 46.27 |
| **500M** | 42.2 | 47.3 | 39.73 |
| **256M** | 33.7 | 40.6 | 32.7 |

---

## 🚀 Getting Started

Install dependencies:

```bash
pip install transformers num2words decord flash-attn
Load the model
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")
```

## Simple Image Inference
```
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```
## Video Inference
```
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "path_to_video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```
## Multi-Image Interleaved Inference
```
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the similarity between these two images?"},
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```
Here is your **clean, perfectly formatted README section for “Training Data”** — ready to paste directly into your README.md.

---

# 📚 Training Data

SmolVLM2 was trained on **3.3 million multimodal samples** sourced from **ten diverse datasets**, including:

* **LlaVa OneVision**
* **M4-Instruct**
* **Mammoth**
* **LLaVa Video 178K**
* **FineVideo**
* **VideoStar**
* **VRipt**
* **Vista-400K**
* **MovieChat**
* **ShareGPT4Video**

These datasets include a mixture of **images**, **videos**, **text-only**, and **multi-image samples**, providing broad visual and linguistic coverage.

---

##  Data Split by Modality

| **Data Type** | **Percentage** |
| ------------- | -------------- |
| Image         | 34.4%          |
| Text          | 20.2%          |
| Video         | 33.0%          |
| Multi-image   | 12.3%          |

---

##  Granular Dataset Distribution

###  Text Datasets

| Dataset                                 | Percentage |
| --------------------------------------- | ---------- |
| llava-onevision/magpie_pro_ft3_80b_mt   | 6.8%       |
| llava-onevision/magpie_pro_ft3_80b_tt   | 6.8%       |
| llava-onevision/magpie_pro_qwen2_72b_tt | 5.8%       |
| llava-onevision/mathqa                  | 0.9%       |

---

###  Multi-Image Datasets

| Dataset                                 | Percentage |
| --------------------------------------- | ---------- |
| m4-instruct-data/m4_instruct_multiimage | 10.4%      |
| mammoth/multiimage-cap6                 | 1.9%       |

---

###  Image Datasets

| Dataset                              | Percentage |
| ------------------------------------ | ---------- |
| llava-onevision/other                | 17.4%      |
| llava-onevision/vision_flan          | 3.9%       |
| llava-onevision/mavis_math_metagen   | 2.6%       |
| llava-onevision/mavis_math_rule_geo  | 2.5%       |
| llava-onevision/sharegpt4o           | 1.7%       |
| llava-onevision/sharegpt4v_coco      | 1.5%       |
| llava-onevision/image_textualization | 1.3%       |
| llava-onevision/sharegpt4v_llava     | 0.9%       |
| llava-onevision/mapqa                | 0.9%       |
| llava-onevision/qa                   | 0.8%       |
| llava-onevision/textocr              | 0.8%       |

---

### Video Datasets

| Dataset                | Percentage |
| ---------------------- | ---------- |
| llava-video-178k/1-2m  | 7.3%       |
| llava-video-178k/2-3m  | 7.0%       |
| other-video/combined   | 5.7%       |
| llava-video-178k/hound | 4.4%       |
| llava-video-178k/0-30s | 2.4%       |
| video-star/starb       | 2.2%       |
| vista-400k/combined    | 2.2%       |
| vript/long             | 1.0%       |
| ShareGPT4Video/all     | 0.8%       |

---

