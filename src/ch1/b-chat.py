from langchain_core.messages import HumanMessage
from src.model import get_model_together

model = get_model_together()

prompt = [HumanMessage("What is the capital of Viet Nam?")]

response = model.invoke(prompt)
print(response.content)