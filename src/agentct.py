import uuid
import adalflow as adal
# Import from specific submodules to avoid ImportError
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter
from adalflow.core.types import GeneratorOutput 
from adalflow.optim.parameter import ParameterType
from adalflow.core import types

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


# 1. The Template: Combines System Prompt + Few Shot Demos + User Input
PROMPT_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}

{% if few_shot_demos %}
Here are some examples of how to think:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
Question: {{input_str}}
<END_OF_USER>
"""

class ObjectCountStudent(Component):
    def __init__(self, client):
        super().__init__()
        self.client = client
        
        # PARAMETER 1: The Instructions
        self.system_prompt = Parameter(
            data="Count the objects in the text below. List all valid objects first before giving the total.",
            role_desc="Task Instructions",
            requires_opt=True,
            param_type=ParameterType.PROMPT
        )
        
        # PARAMETER 2: The Examples (Few-Shot)
        # We initialize it with one simple example to get started.
        self.few_shot_demos = Parameter(
            data="Q: I have a chair and a table. How many items?\nA: Valid objects: Chair, Table.\nTotal count: 2",
            role_desc="Few Shot Examples",
            requires_opt=True,
            param_type=ParameterType.DEMOS
        )

    def call(self, input_str: str):
        # 2. Render the template manually (Robust)
        template = Template(PROMPT_TEMPLATE)
        rendered = template.render(
            system_prompt=self.system_prompt.data,
            few_shot_demos=self.few_shot_demos.data,
            input_str=input_str
        )
        
        # 3. Call Client
        response_text = self.client.call(api_kwargs={"input_str": rendered})
        
        return GeneratorOutput(
            id=str(uuid.uuid4()), 
            data=response_text
        )