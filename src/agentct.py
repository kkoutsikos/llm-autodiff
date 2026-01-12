import uuid
import adalflow as adal
from jinja2 import Template
# Import from specific submodules to avoid ImportError
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter
from adalflow.core.types import GeneratorOutput
from adalflow.optim.parameter import ParameterType
from adalflow.core import types
from typing import Dict, Union, Optional



# ========================================================
# 1. THE TEMPLATE (Instructions + Reasoning + Examples)
# ========================================================
TASK_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}

{# REASONING STRATEGY (New) #}
Reasoning Strategy:
{{reasoning_strategy}}

{# FEW SHOT DEMOS (Uncommented) #}
{% if few_shot_demos %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
Question: {{input_str}}
<END_OF_USER>
"""

# ========================================================
# 2. THE PIPELINE COMPONENT
# ========================================================
class ObjectCountTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        # PARAM 1: Task Instructions
        self.system_prompt = adal.Parameter(
            data="Count the objects in the text below. Provide the final count in the format 'Answer: [[N]]'.",
            role_desc="Task Instructions",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )

        # PARAM 2: Reasoning Strategy (Added)
        self.reasoning_strategy = adal.Parameter(
            data="Think step by step. Break the text down into segments and check each one.",
            role_desc="Reasoning Strategy",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )

        # PARAM 3: Few Shot Demos (Enabled)
        self.few_shot_demos = adal.Parameter(
            data="Q: I have a chair and a table.\nA: Step 1: Found 'chair'. Step 2: Found 'table'. Total: 2.\nAnswer: [[2]]",
            role_desc="Few Shot Examples",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        # The Generator (LLM Wrapper)
        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=TASK_TEMPLATE,
            prompt_kwargs={
                "system_prompt": self.system_prompt,
                "reasoning_strategy": self.reasoning_strategy,
                "few_shot_demos": self.few_shot_demos,
            },
            # CRITICAL: We remove output_processors here.
            # We want raw text so the Teacher can critique formatting errors.
            use_cache=True,
        )

    def call(self, input_str: str, id: str = None) -> adal.GeneratorOutput:
        # Pass the input to the generator
        output = self.llm_counter(prompt_kwargs={"input_str": input_str}, id=id)
        return output

# ========================================================
# 3. ALIAS FOR COMPATIBILITY
# ========================================================
# This allows our training script to import 'ObjectCountStudent' 
# even though we renamed the class to the official name.
ObjectCountStudent = ObjectCountTaskPipeline