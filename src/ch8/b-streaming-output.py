import ast

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_together import ChatTogether
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph import StateGraph, START, END, MessagesState

from dotenv import load_dotenv
import os

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

model = ChatTogether(
    api_key=TOGETHER_API_KEY,
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.1)


@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

searcher = DuckDuckGoSearchRun()
tools = [searcher, calculator]

embeddings = HuggingFaceEmbeddings()

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={"name": tool.name}) for tool in tools ],
    embedding=embeddings,
).as_retriever()

class State(MessagesState):
    selected_tools: list[str]

def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}


def select_node(state: State) -> State:
    query = state["messages"][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}


builder = StateGraph(State)
builder.add_node("select_node", select_node)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "select_node")
builder.add_edge("select_node", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

for c in graph.stream(input, stream_mode="updates"):
    print(c)