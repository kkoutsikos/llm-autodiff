
"""
Local LLM Client Infrastructure.
"""

import logging
import torch
import re
from typing import Any, Dict, Optional, List

# Third-party libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# AdalFlow core imports
from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput, ModelType

# Configure logger
log = logging.getLogger(__name__)

class LocalLLMClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        log.info(f"Loading {self.model_name} with BitsAndBytes NF4 config...")
        
        # FIX: Use float16 for Colab T4 compatibility (bfloat16 requires Ampere GPUs)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            log.error(f"Failed to load model {self.model_name}: {e}")
            raise e

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_args = model_kwargs.copy()
        final_args["input_str"] = input
        return final_args

    def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
        """
        Parses the raw response and cleans XML tags from the optimizer.
        """
        try:
            response_text = str(completion)

            # CLEANING: Remove <proposed_variable> tags if present
            if "<proposed_variable>" in response_text:
                match = re.search(r"<proposed_variable>(.*?)</proposed_variable>", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()

            return GeneratorOutput(data=response_text, raw_response=str(completion))
        except Exception as e:
            return GeneratorOutput(data=None, error=str(e))

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED) -> str:
        # Extract parameters
        system_prompt = api_kwargs.get("system_prompt", None)
        user_input = api_kwargs.get("input_str", "")
        gen_kwargs = api_kwargs.get("model_kwargs", {})

        messages = []
        
        # 1. Handle explicit messages (from Optimizer)
        if "messages" in api_kwargs:
            messages = api_kwargs["messages"]
        # 2. Handle simple input (from Student)
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_input:
                messages.append({"role": "user", "content": user_input})

        if not messages:
            return "" 

        try:
            # Apply Chat Template
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

            max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
            temperature = gen_kwargs.get("temperature", 0.7)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # DECODING: Slice off the input prompt to get ONLY the new text
            input_length = model_inputs.input_ids.shape[1]
            generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
            response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return response_text

        except Exception as e:
            log.error(f"Generation failed: {e}")
            return ""