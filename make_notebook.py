import nbformat as nbf
import os, json

nb = nbf.v4.new_notebook()

# Add proper Jupyter kernelspec metadata for Kaggle / Papermill
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.12"
    }
}

# Cell 1: Header Markdown
cell1 = nbf.v4.new_markdown_cell("""# RGSP - Remote Sensing Vision-Language Model Pipeline Test
**Nghiên cứu:** Dự đoán có chọn lọc dựa trên khả năng phân giải cho MLLM ảnh viễn thám (RGSP - IEEE TGRS).

Notebook này thử nghiệm **pipeline suy luận hoàn chỉnh** trên **1 ảnh viễn thám thật (DOTA)** với **2 mô hình VLM tiêu biểu trong bài báo**:
1. **Qwen2-VL-7B-Instruct** (Đại diện họ Qwen-VL: cơ chế nén dynamic resolution, SigLIP Vision Encoder)
2. **LLaVA-1.5-7B-hf** (Baseline kinh điển: CLIP ViT-L/14@336)

Mỗi mô hình được chạy **2 lần**:
- **Lần 1:** Prompt tiếng Anh (English prompt)
- **Lần 2:** Prompt tiếng Việt (Vietnamese prompt)

> **Cấu hình GPU khuyến nghị trên Kaggle:** Chọn Accelerator: **GPU T4 x 2** hoặc **GPU P100**. Bật **Internet: On**.
""")

# Cell 2: Install dependencies
cell2 = nbf.v4.new_code_cell("""# Cài đặt thư viện cần thiết
!pip install -q --upgrade pip
!pip install -q transformers accelerate bitsandbytes torchvision qwen-vl-utils pillow requests matplotlib

import torch
import gc
from PIL import Image
import requests
import matplotlib.pyplot as plt

print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
""")

# Cell 3: Download & Display Sample Remote Sensing Image
cell3 = nbf.v4.new_code_cell("""# Tải 1 ảnh viễn thám mẫu từ bộ dữ liệu DOTA (nguồn chính của nghiên cứu)
IMAGE_URL = "https://raw.githubusercontent.com/open-mmlab/mmrotate/main/demo/dota_demo.jpg"

image_path = "sample_dota.jpg"
resp = requests.get(IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
with open(image_path, "wb") as f:
    f.write(resp.content)

raw_image = Image.open(image_path).convert("RGB")
print(f"Ảnh viễn thám gốc: {raw_image.size[0]} x {raw_image.size[1]} pixels")

# Resize ảnh nhẹ để chống OOM khi chạy trên GPU T4 (15GB)
resized_image_path = "sample_dota_resized.jpg"
raw_image.thumbnail((768, 768))
raw_image.save(resized_image_path)
print(f"Ảnh sau khi tối ưu hiển thị: {raw_image.size[0]} x {raw_image.size[1]} pixels")

# Hiển thị ảnh
plt.figure(figsize=(10, 6))
plt.imshow(raw_image)
plt.title(f"DOTA Sample Aerial Image ({raw_image.size[0]}x{raw_image.size[1]} px)")
plt.axis("off")
plt.show()
""")

# Cell 4: Define Prompts
cell4 = nbf.v4.new_code_cell('''# Định nghĩa 2 Prompt: Tiếng Anh và Tiếng Việt theo cấu trúc đề tài RGSP
PROMPT_EN = """Examine this aerial remote sensing image carefully:
1. What types of objects or vehicles are visible (e.g., planes, vehicles, ships, harbors, buildings)?
2. Where are they located and what are their orientations?
3. If any object is too blurry or small to determine reliably, state 'Cannot be determined from image'."""

PROMPT_VI = """Quan sát kỹ bức ảnh viễn thám này và trả lời các câu hỏi sau:
1. Trong ảnh có những loại phương tiện hoặc công trình nào (ví dụ: máy bay, ô tô, tàu thuyền, cầu cảng, nhà xưởng)?
2. Vị trí và hướng xoay (trục định hướng) của chúng như thế nào?
3. Nếu có vật thể nào quá nhỏ hoặc quá mờ không thể nhận diện một cách tin cậy, hãy nói rõ 'Không thể xác định từ ảnh'."""

print("--- PROMPT TIẾNG ANH ---")
print(PROMPT_EN)
print("\\n--- PROMPT TIẾNG VIỆT ---")
print(PROMPT_VI)
''')

# Cell 5: Model 1 - Qwen2-VL-7B-Instruct
cell5 = nbf.v4.new_code_cell("""# ==========================================
# 1. MODEL 1: Qwen2-VL-7B-Instruct (4-bit)
# ==========================================
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

qwen_id = "Qwen/Qwen2-VL-7B-Instruct"
print(f"Đang tải mô hình {qwen_id} (chế độ lượng tử hóa 4-bit)...\\n")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Đặt trần max_pixels theo đúng thiết kế của bài báo để kiểm soát số token và chống tràn VRAM
min_pixels = 256 * 28 * 28
max_pixels = 512 * 512

processor_qwen = AutoProcessor.from_pretrained(
    qwen_id,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

model_qwen = Qwen2VLForConditionalGeneration.from_pretrained(
    qwen_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def run_qwen(prompt_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    text = processor_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor_qwen(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        generated_ids = model_qwen.generate(**inputs, max_new_tokens=256)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor_qwen.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Chạy Lần 1: Tiếng Anh
print("\\n--- [QWEN2-VL] CHẠY LẦN 1: PROMPT TIẾNG ANH ---")
out_qwen_en = run_qwen(PROMPT_EN)
print(out_qwen_en)

# Chạy Lần 2: Tiếng Việt
print("\\n--- [QWEN2-VL] CHẠY LẦN 2: PROMPT TIẾNG VIỆT ---")
out_qwen_vi = run_qwen(PROMPT_VI)
print(out_qwen_vi)

# Giải phóng bộ nhớ GPU trước khi nạp model thứ 2
del model_qwen
del processor_qwen
torch.cuda.empty_cache()
gc.collect()
print("\\nĐã giải phóng VRAM của Qwen2-VL!")
""")

# Cell 6: Model 2 - LLaVA-1.5-7B-hf
cell6 = nbf.v4.new_code_cell("""# ==========================================
# 2. MODEL 2: LLaVA-1.5-7B-hf (4-bit)
# ==========================================
from transformers import LlavaForConditionalGeneration, AutoProcessor

llava_id = "llava-hf/llava-1.5-7b-hf"
print(f"Đang tải mô hình {llava_id} (chế độ 4-bit)...\\n")

processor_llava = AutoProcessor.from_pretrained(llava_id)
model_llava = LlavaForConditionalGeneration.from_pretrained(
    llava_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def run_llava(prompt_text):
    prompt = f"USER: <image>\\n{prompt_text}\\nASSISTANT:"
    inputs = processor_llava(text=prompt, images=raw_image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generate_ids = model_llava.generate(**inputs, max_new_tokens=256)
        
    return processor_llava.batch_decode(
        generate_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

# Chạy Lần 1: Tiếng Anh
print("\\n--- [LLAVA-1.5] CHẠY LẦN 1: PROMPT TIẾNG ANH ---")
out_llava_en = run_llava(PROMPT_EN)
print(out_llava_en)

# Chạy Lần 2: Tiếng Việt
print("\\n--- [LLAVA-1.5] CHẠY LẦN 2: PROMPT TIẾNG VIỆT ---")
out_llava_vi = run_llava(PROMPT_VI)
print(out_llava_vi)

# Giải phóng bộ nhớ GPU
del model_llava
del processor_llava
torch.cuda.empty_cache()
gc.collect()
print("\\nĐã giải phóng VRAM của LLaVA-1.5!")
""")

# Cell 7: Summary Comparison
cell7 = nbf.v4.new_code_cell("""# Tổng kết và đối chiếu kết quả 2 mô hình x 2 ngôn ngữ
print("="*80)
print("TỔNG HỢP KẾT QUẢ SO SÁNH 2 MÔ HÌNH VLM TRÊN 1 ẢNH VIỄN THÁM")
print("="*80)

print("\\n1. [QWEN2-VL-7B] TIẾNG ANH:")
print(out_qwen_en)

print("\\n2. [QWEN2-VL-7B] TIẾNG VIỆT:")
print(out_qwen_vi)

print("\\n" + "-"*50)

print("\\n3. [LLAVA-1.5-7B] TIẾNG ANH:")
print(out_llava_en)

print("\\n4. [LLAVA-1.5-7B] TIẾNG VIỆT:")
print(out_llava_vi)
""")

nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7]

kernel_path = r'C:\Users\trang\Desktop\rgsp_nckh\kaggle_kernel\pipeline_test.ipynb'
local_copy = r'C:\Users\trang\Desktop\rgsp_nckh\pipeline_test_vlm.ipynb'

with open(kernel_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

with open(local_copy, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Successfully generated notebook version 3!')
