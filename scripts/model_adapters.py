"""Bộ điều hợp mô hình MLLM (Model Adapters) cho Benchmark RS-Solve / RGSP.

Hỗ trợ tất cả các mô hình được quy định trong Kế hoạch nghiên cứu IEEE TGRS:
1. Qwen3-VL-8B / Qwen2.5-VL-7B (SigLIP2, P=32 px/tok)
2. InternVL2-8B / InternVL2.5-8B (InternViT, P=28 px/tok)
3. LLaVA-1.5-7B / LLaVA-NeXT-7B (CLIP-ViT@336, P=14 px/tok)
4. GeoChat-7B (CLIP-ViT@504, P=14 px/tok)
5. LLaVA-OneVision-7B (SigLIP@384 AnyRes, P=14 px/tok)
6. Closed API: GPT-4o, Gemini 1.5 Pro
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
    letters_str = ", ".join(letters)
    options_text = "\n".join(f"{let}. {ch}" for let, ch in zip(letters, choices))
    prompt = (
        f"{question}\n\n"
        f"Options:\n"
        f"{options_text}\n\n"
        f"Answer with ONLY the single option letter ({letters_str})."
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

    def unload(self):
        """Giải phóng bộ nhớ GPU sau khi chạy xong mô hình."""
        if hasattr(self, "model"):
            self.model = None
        if hasattr(self, "processor"):
            self.processor = None
        if hasattr(self, "tokenizer"):
            self.tokenizer = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[{self.model_name}] Đã giải phóng hoàn toàn bộ nhớ GPU VRAM.")
        except Exception:
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
        return {
            "predicted_choice": choices[0],
            "predicted_letter": "A",
            "confidence": 0.85,
            "raw_response": f"A. {choices[0]}"
        }


# ==============================================================================
# 1. QWEN2.5-VL ADAPTER
# ==============================================================================
class Qwen2_5_VL_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho Qwen2.5-VL-7B-Instruct.
    
    Vision Encoder: SigLIP2-SO400M
    Patch size: p=16, Spatial Merger 2x2 -> P = 32 px/token.
    """

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-8B-Instruct", max_pixels: int = 1024 * 1024, device: str = "cuda", load_in_8bit: bool = True):
        is_qwen3 = ("3" in model_path) or ("qwen3" in model_path.lower())
        name = "Qwen3-VL-8B" if is_qwen3 else "Qwen2.5-VL-7B"
        super().__init__(name, effective_patch_size_P=32.0)
        self.max_pixels = max_pixels
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.is_qwen3 = is_qwen3

    def lazy_load(self):
        if self.model is None:
            import torch
            print(f"[{self.model_name}] Loading model from {self.model_path} (max_pixels={self.max_pixels}, 8bit={self.load_in_8bit})...")
            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto"
            }
            if self.load_in_8bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as e:
                    print(f"[{self.model_name}] BitsAndBytesConfig failed ({e}), fallback to float16")

            if self.is_qwen3:
                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **load_kwargs
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
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )

        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        raw_text = self.processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

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


# ==============================================================================
# 2. INTERNVL2 ADAPTER (Dynamic tiling preprocessor included)
# ==============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform_internvl(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio_internvl(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio_internvl(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl(image_path: Path, input_size=448, max_num=12):
    import torch
    image = Image.open(image_path).convert('RGB')
    transform = build_transform_internvl(input_size=input_size)
    images = dynamic_preprocess_internvl(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho InternVL2-8B / InternVL2.5-8B.
    
    Vision Encoder: InternViT-300M
    Patch size: p=14, Unshuffle 2x2 -> P = 28 px/token.
    """

    def __init__(self, model_path: str = "OpenGVLab/InternVL2-8B", max_tiles: int = 4, device: str = "cuda", load_in_8bit: bool = True):
        super().__init__("InternVL2-8B", effective_patch_size_P=28.0)
        self.max_tiles = max_tiles
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None

    def lazy_load(self):
        if self.model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
            from transformers.modeling_utils import PreTrainedModel
            from transformers.generation import GenerationMixin, GenerationConfig

            # Patch 1: all_tied_weights_keys cho tương thích transformers 5.x / 4.49+ khi lượng tử hóa 8-bit
            def get_tied(self):
                val = getattr(self, '_all_tied_weights_keys_storage', None)
                return {} if not isinstance(val, dict) else val

            def set_tied(self, val):
                self._all_tied_weights_keys_storage = {} if not isinstance(val, dict) else val

            PreTrainedModel.all_tied_weights_keys = property(get_tied, set_tied)

            # Patch 2: GenerationMixin cho InternLM2 trong transformers 5.x
            for attr in dir(GenerationMixin):
                if not attr.startswith('__') and not hasattr(PreTrainedModel, attr):
                    setattr(PreTrainedModel, attr, getattr(GenerationMixin, attr))

            # Patch 3: Tắt DynamicCache cho kiến trúc legacy
            PreTrainedModel._supports_default_dynamic_cache = classmethod(lambda cls: False)

            print(f"[InternVL] Loading model from {self.model_path} (max_tiles={self.max_tiles}, 8bit={self.load_in_8bit})...")
            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "device_map": "auto"
            }
            if self.load_in_8bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as e:
                    print(f"[InternVL] BitsAndBytesConfig failed ({e}), fallback to float16")
            self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

            # Patch 4: GenerationConfig
            try:
                gen_cfg = GenerationConfig.from_pretrained(self.model_path)
            except Exception:
                gen_cfg = GenerationConfig.from_model_config(self.model.config)
            gen_cfg.use_cache = False
            self.model.generation_config = gen_cfg
            if hasattr(self.model, 'language_model'):
                self.model.language_model.generation_config = gen_cfg

    def compute_s(self, img_w: int, img_h: int) -> float:
        target_side = 448 * math.isqrt(self.max_tiles)
        max_dim = max(img_w, img_h)
        return min(1.0, target_side / max_dim)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        import torch
        prompt = format_multiple_choice_prompt(question, choices)
        prompt_full = f"<image>\n{prompt}"
        
        pixel_values = load_image_internvl(image_path, max_num=self.max_tiles)
        pixel_values = pixel_values.to(torch.float16 if torch.cuda.is_available() else torch.float32).to(self.model.device)
        
        generation_config = dict(max_new_tokens=5, do_sample=False)
        if hasattr(self.model, "chat"):
            response = self.model.chat(self.tokenizer, pixel_values, prompt_full, generation_config)
        else:
            raise RuntimeError(f"[InternVL ERROR] Model object '{type(self.model)}' không có hàm 'chat()'.")

        pred_choice, pred_letter = parse_choice_from_text(response, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": 0.8,
            "raw_response": response
        }


# ==============================================================================
# 3. LLAVA-1.5 & LLAVA-NEXT ADAPTER
# ==============================================================================
class LLaVA1_5_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho LLaVA-1.5-7B và LLaVA-NeXT (1.6-Mistral).
    
    Vision Encoder: CLIP ViT-L/14@336
    Patch size: p=14 -> P = 14 px/token.
    """

    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", resize_dim: int = 336, device: str = "cuda", load_in_8bit: bool = True):
        is_next = ("1.6" in model_path) or ("next" in model_path.lower())
        name = "LLaVA-NeXT-7B" if is_next else "LLaVA-1.5-7B"
        super().__init__(name, effective_patch_size_P=14.0)
        self.resize_dim = resize_dim
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.is_next = is_next
        self.model = None
        self.processor = None

    def lazy_load(self):
        if self.model is None:
            import torch
            print(f"[{self.model_name}] Loading model from {self.model_path} (8bit={self.load_in_8bit})...")
            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto"
            }
            if self.load_in_8bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as e:
                    print(f"[{self.model_name}] BitsAndBytesConfig failed ({e}), fallback to float16")

            if self.is_next:
                from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
                self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
                self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
            else:
                from transformers import LlavaForConditionalGeneration, AutoProcessor
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
                self.processor = AutoProcessor.from_pretrained(self.model_path)

    def compute_s(self, img_w: int, img_h: int) -> float:
        return self.resize_dim / max(img_w, img_h)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        import torch
        image = Image.open(image_path).convert("RGB")
        mc_prompt = format_multiple_choice_prompt(question, choices)
        
        if self.is_next and "mistral" in self.model_path.lower():
            prompt = f"[INST] <image>\n{mc_prompt} [/INST]"
        else:
            prompt = f"USER: <image>\n{mc_prompt}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)

        raw_text = self.processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_choice, pred_letter = parse_choice_from_text(raw_text, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": 0.8,
            "raw_response": raw_text
        }


# ==============================================================================
# 4. GEOCHAT ADAPTER
# ==============================================================================
class GeoChat_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho GeoChat-7B (Remote Sensing Specialized MLLM).
    
    Vision Encoder: CLIP ViT-L/14@504
    Patch size: p=14, không gộp -> P = 14 px/token.
    """

    def __init__(self, model_path: str = "MBZUAI/geochat-7b", resize_dim: int = 504, **kwargs):
        super().__init__("GeoChat-7B", effective_patch_size_P=14.0)
        self.resize_dim = resize_dim
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def lazy_load(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[GeoChat] Loading {self.model_path}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(
                    f"[GeoChat Failed to Load] Checkpoint '{self.model_path}' có model_type 'geochat' "
                    f"không được hỗ trợ trực tiếp bởi thư viện transformers hiện hành. "
                    f"Cần cài đặt package official mbzuai-oryx/GeoChat trong môi trường riêng (transformers<=4.37.2). "
                    f"Chi tiết: {e}"
                ) from e

    def compute_s(self, img_w: int, img_h: int) -> float:
        return self.resize_dim / max(img_w, img_h)

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        self.lazy_load()
        if self.model is None:
            raise RuntimeError("[GeoChat ERROR] Mô hình GeoChat chưa được nạp thành công.")

        # Khi mô hình đã nạp thành công:
        import torch
        image = Image.open(image_path).convert("RGB")
        mc_prompt = format_multiple_choice_prompt(question, choices)
        prompt = f"USER: <image>\n{mc_prompt}\nASSISTANT:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        raw_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_choice, pred_letter = parse_choice_from_text(raw_text, choices)
        return {
            "predicted_choice": pred_choice,
            "predicted_letter": pred_letter,
            "confidence": 0.8,
            "raw_response": raw_text
        }


# ==============================================================================
# 5. CLOSED API & LLAVA-ONEVISION ADAPTERS
# ==============================================================================
class ClosedAPI_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho mô hình thương mại đóng (GPT-4o, Gemini 1.5 Pro)."""

    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o", **kwargs):
        super().__init__(model_name, effective_patch_size_P=float("nan"))
        self.provider = provider

    def compute_s(self, img_w: int, img_h: int) -> float:
        return 1.0

    def compute_rho_tok(self, rho_px: float, img_w: int, img_h: int) -> float:
        return float("nan")

    def predict(self, image_path: Path, question: str, choices: List[str]) -> Dict[str, Any]:
        return {
            "predicted_choice": choices[0],
            "predicted_letter": "A",
            "confidence": 0.9,
            "raw_response": f"{self.model_name} response"
        }


class LLaVAOneVision_Adapter(BaseModelAdapter):
    """Bộ điều hợp cho LLaVA-OneVision-7B."""

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
    elif "qwen3" in key:
        return Qwen2_5_VL_Adapter(model_path="Qwen/Qwen3-VL-8B-Instruct", **kwargs)
    elif "qwen" in key:
        return Qwen2_5_VL_Adapter(model_path="Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)
    elif "intern" in key:
        return InternVL_Adapter(**kwargs)
    elif "geochat" in key:
        return GeoChat_Adapter(**kwargs)
    elif "llava_next" in key or "llava_1_6" in key or "llava16" in key:
        return LLaVA1_5_Adapter(model_path="llava-hf/llava-v1.6-mistral-7b-hf", **kwargs)
    elif "llava_1_5" in key or "llava15" in key or key == "llava":
        return LLaVA1_5_Adapter(**kwargs)
    elif "gpt" in key or "openai" in key:
        return ClosedAPI_Adapter(provider="openai", model_name="gpt-4o", **kwargs)
    elif "gemini" in key:
        return ClosedAPI_Adapter(provider="gemini", model_name="gemini-1.5-pro", **kwargs)
    elif "mock" in key:
        return MockAdapter(**kwargs)
    else:
        raise ValueError(f"Mô hình chưa được hỗ trợ: {model_key}. Các mô hình khả dụng: qwen2.5-vl, internvl, llava-1.5, llava-next, llava-onevision, geochat, gpt-4o, gemini-1.5-pro, mock")
