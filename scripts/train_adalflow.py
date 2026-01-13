import sys
import os
sys.path.append(os.getcwd())

import adalflow as adal
from adalflow.core.generator import BackwardPassSetup
from src.client import LocalLLMClient
from src.component import ObjectCountAdalComponent
from src.utils import load_bbh_object_count

def train():
    print("🚀 Starting AdalFlow Trainer...")

    # 1. SETUP CLIENTS
    
    qwen_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct") 
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    
    qwen_config = {
        "model_client": qwen_client,
        "model_kwargs": {"temperature": 0.7}
    }
    
    student_config = {
        "model_client": student_client,
        "model_kwargs": {"temperature": 0.5}
    }

    # 2. INITIALIZE COMPONENT
    adal_component = ObjectCountAdalComponent(
        **student_config, # The Student
        teacher_model_config=qwen_config,       # The Optimizer
        text_optimizer_model_config=qwen_config, # The Updater
        backward_engine_model_config=qwen_config # The Critic
    )
    
    print("\n📦 Component Initialized:")
    print(adal_component)

    # 3. CONFIGURE TRAINER (TextGrad Mode)
    # We enable 'tg=True' to use TextGrad logic
    backward_pass_setup = BackwardPassSetup(
        all_pred_at_once=False, 
        compute_grad_for_errors_only=True
    )

    trainer = adal.Trainer(
        train_batch_size=4,
        adaltask=adal_component,
        strategy="constrained", 
        max_steps=6,           
        num_workers=1,          
        text_optimizers_config_kwargs={
            "max_past_history": 3, 
        },
        backward_pass_setup=backward_pass_setup,
    )
    
    print("\n🏋️ Trainer Configured. Loading Data...")

    # 4. LOAD DATA
    trainset, valset, testset = load_bbh_object_count(n=50)

    # 5. RUN TRAINING
    print("\n🔥 STARTING FIT LOOP...")
    ckpt, _ = trainer.fit(
        train_dataset=trainset,
        val_dataset=valset,
        test_dataset=testset,
    )
    
    print(f"\n✅ Training Complete. Best Checkpoint: {ckpt}")

if __name__ == "__main__":
    train()