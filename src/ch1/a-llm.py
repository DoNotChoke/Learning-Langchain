from langchain_together.chat_models import ChatTogether
from src.model import get_model_together

model = get_model_together()

response = model.invoke("The sky is")
print(response.content)