from langchain_core.prompts import ChatPromptTemplate
from src.model import get_model_together

model = get_model_together()
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

chatbot = template | model

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)


for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part)