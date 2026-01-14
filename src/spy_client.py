
from typing import Dict, Any
from src.client import LocalLLMClient 

class SpyLLMClient(LocalLLMClient):
    def __init__(self, model_name: str, role_name: str = "GENERIC"):
        super().__init__(model_name)
        self.role_name = role_name 

    # --- FIX: Accept **kwargs to handle 'model_type' and other extras ---
    def call(self, api_kwargs: Dict = {}, model_kwargs: Dict = {}, **kwargs):
        
        # 1. Format the Input for Printing
        prompt_display = ""
        if "messages" in api_kwargs:
            for msg in api_kwargs["messages"]:
                role = msg.get('role', 'UNKNOWN').upper()
                content = msg.get('content', '')
                prompt_display += f"\n   [{role}]: {content}"
        else:
            prompt_display = str(api_kwargs)

        print(f"\n🔵 ================== [{self.role_name} INPUT] ==================")
        print(prompt_display.strip())
        print(f"===============================================================\n")

        # 2. Call the Real Model (pass only valid args if needed, or all)
        # LocalLLMClient usually only takes api_kwargs and model_kwargs
        response = super().call(api_kwargs, model_kwargs)

        # 3. Print the Output
        print(f"\n🟢 ------------------ [{self.role_name} RESPONSE] ------------------")
        resp_str = str(response).strip()
        print(resp_str[:500] + "..." if len(resp_str) > 500 else resp_str)
        print(f"---------------------------------------------------------------\n")

        return response