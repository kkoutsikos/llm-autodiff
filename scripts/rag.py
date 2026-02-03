import adalflow as adal
from adalflow.core.types import GeneratorOutput
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.core.generator import BackwardPassSetup
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
import torch
import os
from unittest.mock import MagicMock
import sys
import logging
import gc
import re
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List, Optional

# Free memory
torch.cuda.empty_cache()
gc.collect()

# --- 2. SETUP ---
MOCKS = ["google.generativeai", "anthropic", "groq", "cohere", "mistralai", "boto3", "chromadb", "qdrant_client"]
for mod in MOCKS: sys.modules[mod] = MagicMock()
os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("adalflow").setLevel(logging.ERROR)

@dataclass
class QAExample:
    id: str
    q: str
    a: str

# --- 3. SANITIZED CLIENT ---
class LocalClient(adal.ModelClient):
    def __init__(self, model_name):
        super().__init__()
        print(f"⏳ Loading Model: {model_name}...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def parse_chat_completion(self, completion) -> GeneratorOutput:
        raw = completion["content"]
        content = raw.get("content", str(raw)) if isinstance(raw, dict) else str(raw)
        # Strip Markdown and JSON artifacts to prevent parser crash
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
        return GeneratorOutput(data=content.strip(), raw_response=completion)

    def call(self, api_kwargs={}, model_kwargs={}, **kwargs):
        params = {
            "max_new_tokens": 512, 
            "temperature": model_kwargs.get("temperature", 0.5),
            "return_full_text": False
        }
        msg = [{"role": "user", "content": api_kwargs.get("input", "")}]
        prompt = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        out = self.pipe(prompt, **params)
        return {"content": out[0]["generated_text"]}

# --- 4. RETRIEVER ---
class HotPotRetriever(adal.Component):
    def __init__(self, documents):
        super().__init__()
        self.docs = list(set(documents))
        print(f"📚 Indexing {len(self.docs)} facts...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_vecs = self.embedder.encode(self.docs, show_progress_bar=True)

    def call(self, query: str, id=None):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        if not query or not isinstance(query, str) or len(query) < 2: return self.docs[0]
        q_vec = self.embedder.encode([query])
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        return self.docs[np.argmax(scores)]

    def forward(self, query, id=None):
        text_result = self.call(query, id)
        if self.training:
            return adal.Parameter(data=text_result, requires_opt=False, role_desc="Retrieved Context")
        return text_result

# --- 5. PEER-AWARE TASK ---
class PeerAwareRAG(adal.Component):
    def __init__(self, model_client, model_kwargs, documents):
        super().__init__()
        self.retriever = HotPotRetriever(documents)
        
        # PEER 1: Planner Prompt
        self.planner_prompt = adal.Parameter(
            data="You are a research assistant. Write a specific keyword search query for: ", 
            role_desc="Planner Task Instruction", requires_opt=True, param_type=adal.ParameterType.PROMPT
        )
        self.planner = adal.Generator(
            model_client=model_client, model_kwargs=model_kwargs,
            template="{{system}}{{q}}",
            prompt_kwargs={"system": self.planner_prompt}
        )
        
        # [NEW] PEER 2: Solver Task Instruction (What to do)
        self.solver_task = adal.Parameter(
            data="Answer the question using ONLY the context.", 
            role_desc="Solver Task Instruction", requires_opt=True, param_type=adal.ParameterType.PROMPT
        )
        
        # [NEW] PEER 3: Solver Format Instruction (How to output)
        self.solver_format = adal.Parameter(
            data="Be concise. Do not explain.", 
            role_desc="Solver Output Format", requires_opt=True, param_type=adal.ParameterType.PROMPT
        )
        
        # The Solver uses BOTH peers in its template
        self.solver = adal.Generator(
            model_client=model_client, model_kwargs=model_kwargs,
            template="{{task}} {{format}}\nContext: {{c}}\nQuestion: {{q}}\nAnswer:",
            prompt_kwargs={"task": self.solver_task, "format": self.solver_format}
        )

    def forward(self, question, id=None):
        # 1. Plan
        query_out = self.planner(prompt_kwargs={"q": question}, id=id)
        query_str = query_out.data if isinstance(query_out, adal.Parameter) else query_out.data
        if isinstance(query_str, dict): query_str = query_str.get('content', '')
        
        # 2. Retrieve (Pass-through)
        context = self.retriever(query_str)
        
        # 3. Solve (Using Peers)
        # Note: We don't pass 'task'/'format' here because they are fixed parameters bound in __init__
        answer_out = self.solver(prompt_kwargs={"q": question, "c": context}, id=id)
        return answer_out 

# --- 6. WRAPPER ---
class HotPotQAAdal(adal.AdalComponent):
    def __init__(self, task, text_optimizer_model_config, backward_engine_model_config):
        eval_fn = AnswerMatchAcc(type="f1_score").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(eval_fn=eval_fn, eval_fn_desc="F1 Score")
        super().__init__(
            task=task, 
            eval_fn=eval_fn, 
            loss_fn=loss_fn,
            teacher_model_config=text_optimizer_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            backward_engine_model_config=backward_engine_model_config
        )
    # ... (Boilerplate helpers omitted for brevity, same as before) ...
    def _get_val(self, obj, key):
        if isinstance(obj, dict): return obj.get(key)
        return getattr(obj, key)
    def prepare_task(self, sample): return self.task.forward, {"question": self._get_val(sample, "q"), "id": self._get_val(sample, "id")}
    def prepare_eval(self, sample, y_pred): 
        pred_text = y_pred.data if hasattr(y_pred, 'data') else y_pred
        if isinstance(pred_text, dict): pred_text = pred_text.get('content', '')
        return self.eval_fn, {"y": pred_text, "y_gt": self._get_val(sample, "a")}
    def prepare_loss(self, sample, pred): 
        y_gt_param = adal.Parameter(data=self._get_val(sample, "a"), requires_opt=False)
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt_param}, "id": self._get_val(sample, "id")}

# --- 7. DATA ---
def setup_data_and_docs():
    print("📚 Downloading HotPotQA...")
    raw_data = load_dataset("hotpot_qa", "distractor", split="train[:50]")
    documents = []
    processed_examples = []
    for item in raw_data:
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
             for sent in sentences: documents.append(f"{title}: {sent}")
        processed_examples.append(QAExample(id=item['id'], q=item['question'], a=item['answer']))
    return processed_examples[:10], processed_examples[10:30], documents

# --- 8. TRAIN ---
if __name__ == "__main__":
    print("\n--- LOADING MODELS ---")
    client = LocalClient("Qwen/Qwen2.5-7B-Instruct")
    client2 = LocalClient("Qwen/Qwen2.5-1.5B-Instruct")
    
    student_config = {"model_client": client2, "model_kwargs": {"temperature": 0.5, "max_new_tokens": 1000}}
    teacher_config = {"model_client": client, "model_kwargs": {"temperature": 0.7, "max_new_tokens": 4000}}

    train_d, val_d, documents = setup_data_and_docs()
    
    # Initialize Peer-Aware Task
    print("🏗️ Building Peer-Aware RAG...")
    task = PeerAwareRAG(student_config["model_client"], student_config["model_kwargs"], documents)
    
    # Initialize Wrapper
    adal_component = HotPotQAAdal(task, teacher_config, teacher_config)
    
    # Configure Trainer (Error Only + Constrained)
    # Note: 'constrained' strategy implements the "Two-Stage Validation" described in the paper [cite: 412]
    backward_setup = BackwardPassSetup(all_pred_at_once=False, compute_grad_for_errors_only=True)

    trainer = adal.Trainer(
        train_batch_size=2,
        adaltask=adal_component,
        strategy="constrained",
        max_steps=1,
        backward_pass_setup=backward_setup 
    )
    
    print("\n🔥 STARTING PEER-AWARE OPTIMIZATION 🔥")
    ckpt, _ = trainer.fit(train_dataset=train_d, val_dataset=val_d)
    
    print(f"\n✅ Checkpoint Saved")
    print(f"Updated Solver Task: {trainer.adaltask.task.solver_task.data}")
    print(f"Updated Solver Format: {trainer.adaltask.task.solver_format.data}")