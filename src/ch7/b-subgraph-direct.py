from typing import TypedDict

from langgraph.graph import StateGraph, START

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    foo: str
    bar: str

def subgraph_node(state: SubgraphState):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")

subgraph = subgraph_builder.compile()

builder = StateGraph(State)
builder.add_node("subgraph", subgraph)
builder.add_edge(START, "subgraph")

graph = builder.compile()

initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(f"Result: {result}")