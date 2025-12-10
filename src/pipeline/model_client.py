import logging
import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForImageTextToText
)
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def _clean_omr_response(text):
    if not text:
        return ""

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "M:" in part and "K:" in part:
                text = part.replace("abc", "").strip()
                break

    if "M:" in text:
        text = "M:" + text.split("M:", 1)[1]

    return text.strip()


class ModelClientQwen2VL:
    def __init__(self, model_config):
        self.model_id = model_config['model_id']
        self.trust_remote_code = model_config.get('trust_remote_code', True)
        self.model_kwargs = model_config.get('model_kwargs', {})
        self.gen_kwargs = model_config.get('generate_kwargs', {})

        if 'dtype' in self.model_kwargs:
            if self.model_kwargs['dtype'] == 'bfloat16':
                self.model_kwargs['dtype'] = torch.bfloat16
            elif self.model_kwargs['dtype'] == 'float16':
                self.model_kwargs['dtype'] = torch.float16

        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading processor: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)

        logger.info(f"Loading model: {self.model_id}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            **self.model_kwargs
        ).eval()

        if torch.backends.mps.is_available() and self.model.device.type == 'cpu':
            logger.info("Moving model to MPS (Apple Silicon GPU)...")
            self.model.to('mps')

        if torch.cuda.is_available() and self.model.device.type == 'cpu':
            logger.info("Moving model to CUDA GPU ...")
            self.model.to("cuda")

        logger.info(f"Model loaded successfully on device: {self.model.device}")

    def generate(self, image_path, prompt_text):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, **self.gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            raw_response = output_text[0].strip()
            return _clean_omr_response(raw_response)

        except Exception as e:
            logger.error(f"Generation error for {image_path}: {e}")
            return ""


class ModelClientSmolVLM:
    def __init__(self, model_config):
        self.model_id = model_config['model_id']
        self.trust_remote_code = model_config.get('trust_remote_code', True)
        self.model_kwargs = model_config.get('model_kwargs', {})
        self.gen_kwargs = model_config.get('generate_kwargs', {})

        if 'dtype' in self.model_kwargs:
            if self.model_kwargs['dtype'] == 'bfloat16':
                self.model_kwargs['dtype'] = torch.bfloat16
            elif self.model_kwargs['dtype'] == 'float16':
                self.model_kwargs['dtype'] = torch.float16

        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading processor: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)

        logger.info(f"Loading model: {self.model_id}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            **self.model_kwargs
        ).eval()

        if torch.backends.mps.is_available() and self.model.device.type == 'cpu':
            logger.info("Moving model to MPS (Apple Silicon GPU)...")
            self.model.to('mps')

        if torch.cuda.is_available() and self.model.device.type == 'cpu':
            logger.info("Moving model to CUDA GPU ...")
            self.model.to("cuda")

        logger.info(f"Model loaded successfully on device: {self.model.device}")

    def generate(self, image_path, prompt_text):
        try:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )

            inputs = inputs.to(self.model.device)
            if self.model.dtype == torch.bfloat16:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            generated_ids = self.model.generate(**inputs, **self.gen_kwargs)

            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            raw_response = output_text[0].strip()
            return _clean_omr_response(raw_response)

        except Exception as e:
            logger.error(f"Generation error for {image_path}: {e}")
            return ""


def get_model_client(model_config):
    model_id = model_config['model_id'].lower()

    if "qwen" in model_id:
        return ModelClientQwen2VL(model_config)
    elif "smolvlm" in model_id:
        return ModelClientSmolVLM(model_config)
    else:
        raise ValueError(f"No client implementation found for model: {model_id}")
