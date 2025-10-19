
#  Idefics3-8B

**Idefics3-8B** is a multimodal (image + text) model capable of performing inference on tasks where the input is a combination of text and images — including **image captioning**, **visual question answering**, and other **multimodal reasoning** tasks.  

---

## Overview

Idefics3-8B can process **arbitrarily interleaved text and images**, making it suitable for a variety of multimodal tasks.  

The model was **post-trained via supervised fine-tuning only**, without RLHF alignment.  
As a result, it may sometimes produce short responses or require prompt adjustments to generate complete answers.


---

##  Fine-Tuning

To fine-tune **Idefics3-8B** on your own dataset, check out the provided fine-tuning tutorials.  
The process is similar to fine-tuning **Idefics2**.

**Resources**
-  Fine-tuning with [TRL library (Script)]()
-  Fine-tuning with [Hugging Face Trainer (Tutorial Notebook)]()

---

##  Technical Summary

Idefics3 demonstrates significant improvements over Idefics2, particularly in **document understanding tasks**, and serves as a robust foundation for task-specific fine-tuning.

| Model | MMMU (val) | MathVista (test) | MMStar (val) | DocVQA (test) | TextVQA (val) |
|--------|-------------|------------------|---------------|----------------|----------------|
| **Idefics2-8B** | 45.2 | 52.2 | 49.5 | 74.0 | 73.0 |
| **Idefics3-8B** | 46.6 | 58.4 | 55.9 | 87.7 | 74.9 |

---

##  Model Improvements in Idefics3

-  Uses **169 visual tokens** per image of size **364 × 364**  
  Each image is divided into sub-images (≤ 364 × 364) and encoded separately.  
-  Extended **The Cauldron** fine-tuning dataset with additional sources (e.g., **Docmatix**).  
- Further training details are available in the **technical report** *(coming soon)*.

---

## 💡 How to Get Started

Below is a sample Python script demonstrating inference with **Idefics3-8B**.

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Load images
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")

# Load model and processor
processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3",
    torch_dtype=torch.bfloat16
).to(DEVICE)

# Create messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ]
    },       
]

# Apply chat template
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# Generate response
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
````

---

## 🧮 Text Generation Inference

> **TODO:** Documentation for text generation inference setup and optimization.

---

## ⚙️ Model Optimizations

To optimize for GPU memory and speed:

```python
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3",
    torch_dtype=torch.bfloat16
).to(DEVICE)
```

---

### 🪶 Vision Encoder Efficiency

You can adjust the image scaling by setting:

```python
processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3",
    size={"longest_edge": N*364}
)
```

* `N = 4` (default) works best in most cases
* Increase to `N = 5` for very large images
* Decrease to `N = 2` or `N = 3` if GPU memory is limited

---

## ⚡ Using Flash-Attention 2

Flash-Attention 2 can be used to **speed up generation**.
Configuration instructions will be added soon.

## 📘 References

* [Hugging Face Model Card (Idefics3-8B-Llama3)](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
* [Technical Report (Coming Soon)]()


