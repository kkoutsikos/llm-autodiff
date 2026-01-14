import sys
import os
sys.path.append(os.getcwd())

import adalflow as adal
from adalflow.core.generator import BackwardPassSetup
from src.spy_client import SpyLLMClient  # <--- IMPORT THE SPY
from src.component import ObjectCountAdalComponent
from src.utils import load_bbh_object_count

def train():
    print("🚀 Starting Training with FULL PROMPT LOGGING...")

    # 1. Define 3 "Spy" Clients
    # This lets us distinguish logs in the console
    student_client = SpyLLMClient("Qwen/Qwen2.5-1.5B-Instruct", role_name="STUDENT")
    teacher_client = SpyLLMClient("Qwen/Qwen2.5-7B-Instruct", role_name="TEACHER (CRITIC)")
    optimizer_client = SpyLLMClient("Qwen/Qwen2.5-7B-Instruct", role_name="OPTIMIZER (BRAIN)")

    # 2. Assign them to their roles
    # The Student Config
    student_config = {
        "model_client": student_client,
        "model_kwargs": {"temperature": 0.5}
    }
    
    # The Teacher/Optimizer Configs
    # Note: We use the same 'optimizer_client' for both Teacher roles if we want, 
    # or separate them. Here I separate them for clarity.
    teacher_kwargs = {
        "temperature": 0.7,
        "max_tokens": 4096 
    }

    adal_component = ObjectCountAdalComponent(
        # The Student sees questions
        model_client=student_client, 
        model_kwargs={"temperature": 0.5},
        
        # The Backward Engine sees failures & critiques them
        backward_engine_model_config={
            "model_client": teacher_client, 
            "model_kwargs": teacher_kwargs
        },
        
        # The Text Optimizer sees critiques & writes new prompts
        text_optimizer_model_config={
            "model_client": optimizer_client, 
            "model_kwargs": teacher_kwargs
        },
        
        # (Optional) Teacher model config if used elsewhere
        teacher_model_config={
            "model_client": teacher_client, 
            "model_kwargs": teacher_kwargs
        }
    )

    # 3. Setup Trainer (Reduced history to prevent overflow)
    backward_pass_setup = BackwardPassSetup(
        all_pred_at_once=False,
        compute_grad_for_errors_only=True 
    )

    trainer = adal.Trainer(
        train_batch_size=4,
        adaltask=adal_component,
        strategy="constrained",
        max_steps=5, # Keep it short for testing
        num_workers=1,
        text_optimizers_config_kwargs={
            "max_past_history": 2, 
            
        },
        backward_pass_setup=backward_pass_setup,
    )
    
    trainset, valset, testset = load_bbh_object_count(n=30)

    print("\n🔥 STARTING RUN...")
    trainer.fit(train_dataset=trainset, val_dataset=valset, test_dataset=testset)

if __name__ == "__main__":
    train()