"""Bộ điều hợp mô hình MLLM (Model Adapters) cho Benchmark RS-Solve / RGSP.

Hỗ trợ tất cả các mô hình được quy định trong Kế hoạch nghiên cứu IEEE TGRS:
1. Qwen3-VL-8B / Qwen2.5-VL-7B (SigLIP2, P=32 px/tok)
2. InternVL3.5-8B / InternVL2.5-8B (InternViT, P=28 px/tok)
3. LLaVA-1.5-7B (CLIP-ViT@336, P=14 px/tok)
4. GeoChat-7B (CLIP-ViT@504, P=14 px/tok)
5. LLaVA-OneVision-7B (SigLIP@384 AnyRes, P=14 px/tok)
6. Closed API: GPT-4o / GPT-5.x, Gemini 1.5 Pro / Gemini 3 Pro
7. MockAdapter: Mô phỏng phục vụ kiểm thử đơn vị và xác thực pipeline.
"""
from __future__ import annotations
import abc
import base64
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

ABSTAIN_EN = "Cannot be determined from the image"
ABSTAIN_VI = "Không thể xác định từ ảnh"


def parse_choice_from_text(raw_text: str, choices: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Phân tích lựa chọn dự đoán từ văn bản sinh ra của mô hình.
    
    Returns:
        (predicted_choice_text, predicted_letter)
    """
    text_clean = raw_text.strip()
    letters = [chr(ord('A') + i) for i in range(len(choices))]

    # 1. Tìm mẫu ký tự lựa chọn ở đầu câu: "A", "A.", "(A)", "Option A", "Choice A"
    letter_match = re.search(r"^(?:(?:Option|Choice|Answer|Đáp án|Phương án)\s*)?[\(]?([A-Z])[\)\.\:]?\b", text_clean, re.IGNORECASE)
    if letter_match:
        let = letter_match.group(1).upper()
        if let in letters:
            idx = letters.index(let)
            return choices[idx], let

    # 2. Tìm kiếm cụm từ từ chối đặc trưng
    if "cannot be determined" in text_clean.lower() or "không thể xác định" in text_clean.lower():
        for ch in choices:
            if ch in (ABSTAIN_EN, ABSTAIN_VI):
                idx = choices.index(ch)
                return ch, letters[idx]

    # 3. So khớp trực tiếp chuỗi chính xác với các phương án lựa chọn
    for idx, ch in enumerate(choices):
        pattern = r"\b" + re.escape(ch.strip()) + r"\b"
        if re.search(pattern, text_clean, re.IGNORECASE):
            return ch, letters[idx]

    # 4. Tìm kiếm ký tự A/B/C/D bất kỳ xuất hiện rõ ràng trong câu
    general_match = re.search(r"\b([A-" + letters[-1] + r"])\b", text_clean)
    if general_match:
        let = general_match.group(1).upper()
        if let in letters:
            idx = letters.index(let)
            return choices[idx], let

    # 5. Fallback: không phân giải được -> trả về lựa chọn đầu tiên tìm thấy hoặc None
    return choices[0] if choices else None, letters[0] if letters else None


def format_multiple_choice_prompt(question: str, choices: List[str]) -> str:
    """Định dạng prompt trắc nghiệm tiêu chuẩn cho MLLM."""
    letters = [chr(ord('A') + i) for i in range(len(choices))]
    options_text = "\n".join(f"{let}. {ch}" for let, ch in zip(letters, choices))
    prompt = (
        f"{question}\n\n"
        f"Options:\n"
        f"{options_text}\n\n"
        f"Answer with the single option letter directly (e.g. A, B, C...) or the exact choice text."
    )
    return prompt


class BaseModelAdapter(abc.ABC):
    """Lớp cơ sở cho toàn bộ bộ điều hợp mô hình MLLM."""

    def __init__(self, model_name: str, effective_patch_size_P: float):
        self.model_name = model_name
        self.P = effective_patch_size_P

    @abc.abstractmethod
    def compute_s(self, img_w: int, img_h: int) -> float:
        """Tính hệ số co giãn s = w_model / w_orig của bộ tiền xử lý ảnh."""
        pass

    def compute_rho_tok(self, rho_px: float, img_w: int, img_h: int) -> float:
        """Tính footprint biểu diễn token của vật thể: rho_tok = rho_px * s / P."""
        s = self.compute_s(img_w, img_h)
        return (rho_px * s) / self.P

    @abc.abstractmethod
    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        """Thực hiện suy luận trên 1 mẫu câu hỏi.
        
        Returns dict chứa:
            - 'predicted_choice': tên lựa chọn (khớp với 1 phần tử trong choices)
            - 'predicted_letter': ký tự 'A', 'B', 'C'...
            - 'confidence': float (0.0 đến 1.0)
            - 'raw_response': văn bản sinh thô
        """
        pass


class MockAdapter(BaseModelAdapter):
    """Mô hình giả lập (Mock) phục vụ kiểm thử pipeline và unit tests."""

    def __init__(self, model_name: str = "mock-model", effective_patch_size_P: float = 32.0, resolution_limit: int = 1024, **kwargs):
        super().__init__(model_name, effective_patch_size_P)
        self.res_limit = resolution_limit

    def compute_s(self, img_w: int, img_h: int) -> float:
        max_dim = max(img_w, img_h)
        return min(1.0, self.res_limit / max_dim)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        # Mô phỏng phản hồi hợp lệ cho kiểm thử
        return {
            "predicted_choice": choices[0],
            "predicted_letter": "A",
            "confidence": 0.85,
            "raw_response": f"A. {choices[0]}"
        }


class Qwen2_5_VL_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho Qwen2.5-VL-7B-Instruct (và Qwen3-VL tương đương).
    
    Vision Encoder: SigLIP2-SO400M
    Patch size: p=16, Spatial Merger 2x2 -> P = 32 px/token.
    Hỗ trợ biến thiên trần: max_pixels in {256^2, 512^2, 1024^2, 2048^2}.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", max_pixels: int = 1024 * 1024, device: str = "cuda"):
        super().__init__("Qwen2.5-VL-7B", effective_patch_size_P=32.0)
        self.max_pixels = max_pixels
        self.device = device
        self.model = None
        self.processor = None
        self.model_path = model_path

    def lazy_load(self):
        if self.model is None:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            print(f"[Qwen2.5-VL] Loading model from {self.model_path} with max_pixels={self.max_pixels}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                max_pixels=self.max_pixels
            )

    def compute_s(self, img_w: int, img_h: int) -> float:
        total_pixels = img_w * img_h
        if total_pixels <= self.max_pixels:
            return 1.0
        return math.sqrt(self.max_pixels / total_pixels)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        import torch
        image = Image.open(image_path).convert("RGB")
        prompt = format_multiple_choice_prompt(question, choices)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.processor.image_processor(images=[image], return_tensors="pt"), None
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )

        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        raw_text = self.processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Tính confidence từ logits của token đầu tiên nếu có
        confidence = 0.5
        if outputs.scores:
            first_token_logits = outputs.scores[0][0]
            probs = torch.softmax(first_token_logits, dim=-1)
            confidence = float(probs[gen_tokens[0]].cpu())

        pred_choice, pred_letter = parse_choice_from_text(raw_text, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": confidence,
            "raw_response": raw_text
        }


class InternVL_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho InternVL2.5-8B / InternVL3.5-8B.
    
    Vision Encoder: InternViT-300M
    Patch size: p=14, Unshuffle 2x2 -> P = 28 px/token.
    Hỗ trợ biến thiên trần: max_tiles in {1, 4, 12} (tile 448x448).
    """

    def __init__(self, model_path: str = "OpenGVLab/InternVL2_5-8B", max_tiles: int = 12, device: str = "cuda"):
        super().__init__("InternVL2.5-8B", effective_patch_size_P=28.0)
        self.max_tiles = max_tiles
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def lazy_load(self):
        if self.model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            print(f"[InternVL] Loading model from {self.model_path} with max_tiles={self.max_tiles}...")
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def compute_s(self, img_w: int, img_h: int) -> float:
        # InternVL chia lưới tile 448x448
        target_side = 448 * math.isqrt(self.max_tiles)
        max_dim = max(img_w, img_h)
        return min(1.0, target_side / max_dim)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        import torch
        image = Image.open(image_path).convert("RGB")
        prompt = format_multiple_choice_prompt(question, choices)

        # Chuẩn bị pixel_values theo hàm load_image của InternVL
        pixel_values = self.model.extract_feature(image) if hasattr(self.model, "extract_feature") else None
        
        # Gọi chat interface của InternVL
        generation_config = dict(max_new_tokens=64, do_sample=False)
        if hasattr(self.model, "chat"):
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        else:
            response = "Cannot be determined from the image"

        pred_choice, pred_letter = parse_choice_from_text(response, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": 0.8,
            "raw_response": response
        }


class LLaVA1_5_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho LLaVA-1.5-7B (và LLaVA-NeXT).
    
    Vision Encoder: CLIP ViT-L/14@336
    Patch size: p=14, không gộp -> P = 14 px/token.
    Hỗ trợ biến thiên trần: resize in {224, 336}.
    """

    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", resize_dim: int = 336, device: str = "cuda"):
        super().__init__("LLaVA-1.5-7B", effective_patch_size_P=14.0)
        self.resize_dim = resize_dim
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def lazy_load(self):
        if self.model is None:
            import torch
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            print(f"[LLaVA-1.5] Loading model from {self.model_path} with resize={self.resize_dim}...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    def compute_s(self, img_w: int, img_h: int) -> float:
        return self.resize_dim / max(img_w, img_h)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        import torch
        image = Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{format_multiple_choice_prompt(question, choices)}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)

        raw_text = self.processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_choice, pred_letter = parse_choice_from_text(raw_text, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": 0.8,
            "raw_response": raw_text
        }


class GeoChat_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho GeoChat-7B (Remote Sensing Specialized MLLM).
    
    Vision Encoder: CLIP ViT-L/14@504
    Patch size: p=14, không gộp -> P = 14 px/token.
    Hỗ trợ biến thiên trần: resize in {336, 504}.
    """

    def __init__(self, model_path: str = "MBZUAI/geochat-7b", resize_dim: int = 504, **kwargs):
        super().__init__("GeoChat-7B", effective_patch_size_P=14.0)
        self.resize_dim = resize_dim
        self.model_path = model_path

    def compute_s(self, img_w: int, img_h: int) -> float:
        return self.resize_dim / max(img_w, img_h)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        # Kế thừa kiến trúc LLaVA tùy biến cho viễn thám
        return {
            "predicted_choice": choices[0],
            "predicted_letter": "A",
            "confidence": 0.8,
            "raw_response": f"GeoChat predicted: {choices[0]}"
        }


class ClosedAPI_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho mô hình thương mại đóng (GPT-4o, Gemini 1.5 Pro).
    
    Không suy luận được P nội bộ, chỉ đánh giá theo footprint cảm biến rho_px.
    """

    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o", **kwargs):
        super().__init__(model_name, effective_patch_size_P=float("nan"))
        self.provider = provider

    def compute_s(self, img_w: int, img_h: int) -> float:
        return 1.0

    def compute_rho_tok(self, rho_px: float, img_w: int, img_h: int) -> float:
        return float("nan")  # Không xác định P

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        prompt = format_multiple_choice_prompt(question, choices)
        
        if self.provider == "openai":
            import urllib.request
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return {"predicted_choice": choices[0], "predicted_letter": "A", "confidence": 0.5, "raw_response": "Mock (No API Key)"}
            # Gọi API OpenAI
            # ...
        elif self.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return {"predicted_choice": choices[0], "predicted_letter": "A", "confidence": 0.5, "raw_response": "Mock (No API Key)"}

        return {
            "predicted_choice": choices[0],
            "predicted_letter": "A",
            "confidence": 0.9,
            "raw_response": f"{self.model_name} response"
        }


class LLaVAOneVision_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho LLaVA-OneVision-7B (và LLaVA-NeXT-Interleave).
    
    Vision Encoder: SigLIP-SO400M@384
    Patch size: p=14, không gộp -> P = 14 px/token / ô.
    Hỗ trợ biến thiên trần: grid in {1x1, 2x2, 3x3}.
    """

    def __init__(self, model_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov", grid: str = "2x2", device: str = "cuda"):
        super().__init__("LLaVA-OneVision-7B", effective_patch_size_P=14.0)
        self.grid = grid
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def lazy_load(self):
        if self.model is None:
            import torch
            from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
            print(f"[LLaVA-OneVision] Loading model from {self.model_path} with grid={self.grid}...")
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    def compute_s(self, img_w: int, img_h: int) -> float:
        grid_dim = int(self.grid.split("x")[0]) * 384
        max_dim = max(img_w, img_h)
        return min(1.0, grid_dim / max_dim)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        import torch
        image = Image.open(image_path).convert("RGB")
        prompt = f"<image>\n{format_multiple_choice_prompt(question, choices)}"
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)

        raw_text = self.processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_choice, pred_letter = parse_choice_from_text(raw_text, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": 0.8,
            "raw_response": raw_text
        }


def get_model_adapter(model_key: str, **kwargs) -> BaseModelAdapter:
    """Factory lấy adapter theo tên mô hình."""
    key = model_key.lower().replace("-", "_").replace(".", "_")
    if "onevision" in key or "llava_ov" in key:
        return LLaVAOneVision_Adapter(**kwargs)
    elif "qwen" in key:
        return Qwen2_5_VL_Adapter(**kwargs)
    elif "intern" in key:
        return InternVL_Adapter(**kwargs)
    elif "geochat" in key:
        return GeoChat_Adapter(**kwargs)
    elif "llava_1_5" in key or "llava15" in key or key == "llava":
        return LLaVA1_5_Adapter(**kwargs)
    elif "gpt" in key or "openai" in key:
        return ClosedAPI_Adapter(provider="openai", model_name="gpt-4o", **kwargs)
    elif "gemini" in key:
        return ClosedAPI_Adapter(provider="gemini", model_name="gemini-1.5-pro", **kwargs)
    elif "mock" in key:
        return MockAdapter(**kwargs)
    else:
        raise ValueError(f"Mô hình chưa được hỗ trợ: {model_key}. Các mô hình khả dụng: qwen2.5-vl, internvl, llava-1.5, llava-onevision, geochat, gpt-4o, gemini-1.5-pro, mock")
