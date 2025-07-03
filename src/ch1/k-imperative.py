from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from src.model import get_model

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ]
)

model = get_model()

@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)