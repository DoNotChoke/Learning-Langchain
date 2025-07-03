from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from src.model import get_model

model = get_model()

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ]
)

@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part.content)