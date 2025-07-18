from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

from src.model import get_model_together

model = get_model_together()

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(input):
    print(chunk)