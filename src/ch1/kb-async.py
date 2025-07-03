from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from src.model import get_model
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = get_model()

@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

async def main():
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == '__main__':
    import asyncio
    print(asyncio.run(main()))