from langchain_core.messages import HumanMessage
from src.model import get_model

model = get_model()

prompt = [HumanMessage("What is the capital of Viet Nam?")]

response = model.invoke(prompt)
print(response.content)