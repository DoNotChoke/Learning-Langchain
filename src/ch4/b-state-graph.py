from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.model import get_model_together

class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
model = get_model_together()

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

input = {"messages": [HumanMessage("hi!")]}

for chunk in graph.stream(input):
    print(chunk)