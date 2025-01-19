import json
import openai
from pydantic import BaseModel, Field

class Classification(BaseModel):
    classification : str=Field(description="Classification as either True or False")


class OpenAIServiceClass:
    def __init__(self):
        self.client = openai.OpenAI(api_key="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")

        self.system_prompt_classify_question = """You are a classifier that classifies a message as either health/cooking-related or not. 
        Your task is to output True if user message is related to health/cooking, greetings, general questions etc. 
        Otherwise, you should output False if the message is related to any other TOPIC. Respond in json output. The user message is {message} 
        """
    
    def classify_question(self, message):
        system_prompt = self.system_prompt_classify_question.format(message = message)

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": system_prompt,
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Classification",
                        "description": "A list of KPIs names generated for given role",
                        "parameters": Classification.model_json_schema(),
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "Classification"},
            },
            temperature=0.1,
        )

        print(completion.choices[0].message.tool_calls[0].function.arguments)
        
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments) 

        
