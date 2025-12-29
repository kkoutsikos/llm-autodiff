import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from adalflow.core import ModelClient

# Global Cache to prevent reloading shards between calls
_SHARED_CLIENT = None

class LocalLLMClient(ModelClient):
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        super().__init__()
        print(f"Loading {model_id} into VRAM...")
        
        # 1. Quantization Config (Crucial for 8GB-16GB Cards)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # 2. Load Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 3. Create Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True, # Allow creativity for critiques
            temperature=0.7
        )

    def call(self, api_kwargs={}, model_kwargs={}):
        """Standard AdalFlow Entry Point"""
        prompt = api_kwargs.get("messages", api_kwargs.get("prompt"))
        
        # Apply Chat Template (Qwen/Llama require this)
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        
        # Run Inference
        outputs = self.pipe(prompt)
        return outputs[0]['generated_text'][len(prompt):] # Return only new text

def get_client():
    """Singleton Accessor"""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = LocalLLMClient()
    return _SHARED_CLIENT