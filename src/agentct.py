from typing import Dict, Union
import adalflow as adal
from adalflow.optim.parameter import ParameterType

# =====================================================
# 1. SIMPLE TEMPLATE (No hardcoded headers)
# =====================================================
# We remove "Reasoning Strategy:" and "Here are examples:"
# We let the Optimizer write those headers IF it wants to.
SIMPLE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""

# =====================================================
# 2. THE MONOLITHIC STUDENT
# =====================================================
class ObjectCountTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        # ONE PARAMETER TO RULE THEM ALL
        # We put the instructions, strategy, and examples ALL inside this one variable.
        # This gives the Optimizer 100% freedom to rewrite the entire logic.


        self.system_prompt = adal.Parameter(
            data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            role_desc="Task Instructions",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )

        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=SIMPLE_TEMPLATE,
            prompt_kwargs={
                "system_prompt": self.system_prompt,
            },
            use_cache=True,
        )

    def bicall(self, question: str, id: str = None) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        if output.data and hasattr(output.data, 'error') and output.data.error and "429" in output.data.error:
            raise ValueError("Rate limit exceeded")
        return output

    def __call__(self, question: str, id: str = None):
        return self.bicall(question, id)

# ALIAS
ObjectCountStudent = ObjectCountTaskPipeline