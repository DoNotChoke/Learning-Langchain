from langchain_core.messages import HumanMessage, SystemMessage
from src.model import get_model_together

model = get_model_together()

system_msg = SystemMessage(
    "You are a helpful assistant that responds to human's question with three exclamation marks."
)

human_msg = HumanMessage(
    "What is the capital of Viet Nam?"
)
response = model.invoke([system_msg, human_msg])
print(response.content)