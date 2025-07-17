from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from src.model import get_model_together

class RouterQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question",
    )

llm = get_model_together()
structured_llm = llm.with_structured_output(RouterQuery)

system = """You are an expert at routing a user question to the appropriate data source. Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

router = prompt | structured_llm

question = """Why doesn't the following code work: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french") """

result = router.invoke({"question": question})
print("\nRouting to: ", result)

def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    else:
        return "chain for js_docs"


full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({"question": question})
print("\nChoose route: ", result)