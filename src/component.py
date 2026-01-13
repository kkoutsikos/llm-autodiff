from typing import Any, Callable, Dict, Optional, Tuple
import adalflow as adal
from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from src.agentct import ObjectCountTaskPipeline # Your Student

class ObjectCountAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Optional[Dict] = None,
        teacher_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None,
    ):
        # 1. Initialize the Student Pipeline
        task = ObjectCountTaskPipeline(model_client, model_kwargs)
        
        # 2. Define Eval Function (Exact Match)
        # We assume strict integer matching for simplicity
        def exact_match_acc(y, y_gt):
            # Parse numbers from strings like "Answer: [[5]]"
            import re
            def extract(s):
                if not s: return -999
                m = re.findall(r"\d+", str(s))
                return int(m[-1]) if m else -998
            
            val_y = extract(y)
            val_gt = extract(y_gt)
            return 1.0 if val_y == val_gt else 0.0

        eval_fn = exact_match_acc
        
        # 3. Define Loss Function (Textual Loss)
        # This converts the Eval Score (0 or 1) into a text critique
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if prediction matches ground truth integer, else 0",
        )
        
        super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

        # 4. Configs for the Trainer
        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config

    def prepare_task(self, sample: Example) -> Tuple[Callable, Dict[str, Any]]:
        # Unpacks the dataset sample for the Student
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(
        self, sample: Example, y_pred: adal.GeneratorOutput
    ) -> Tuple[float, Dict[str, Any]]:
        # Unpacks data for the Evaluator
        y_label = ""
        if y_pred and y_pred.data:
            y_label = y_pred.data
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss(
        self, sample: Example, pred: adal.Parameter
    ) -> Tuple[Callable, Dict[str, Any]]:
        # Unpacks data for the Loss Function (Critique Generator)
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )
        # Critical: Pass the raw data for critique
        pred.eval_input = pred.data.data if hasattr(pred.data, 'data') else str(pred.data)
        
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}