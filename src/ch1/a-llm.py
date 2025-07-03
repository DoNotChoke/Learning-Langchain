from langchain_together.chat_models import ChatTogether
from src.model import get_model

model = get_model()

response = model.invoke("The sky is")
print(response.content)