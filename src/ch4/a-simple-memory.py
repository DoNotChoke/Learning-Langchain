from langchain_core.prompts import ChatPromptTemplate
from src.model import get_model_together

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    ("placeholder", "{messages}"),
])

model = get_model_together()

chain = prompt | model

response = chain.invoke({
    "messages": [
        ("human", "Translate this sentence from English to French: I love programming."),
        ("ai", "J'adore programmer."),
        ("human", "What did you just say?"),
    ]
})

print(response.content)

