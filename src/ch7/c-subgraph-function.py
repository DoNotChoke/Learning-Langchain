from typing import TypedDict
from langgraph.graph import StateGraph, START

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    bar: str
    baz: str

def subgraph_node(state: SubgraphState):
    return {"bar": state["bar"] + "baz"}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")

subgraph = subgraph_builder.compile()

def node(state: State):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}

builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")

graph = builder.compile()


initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(
    f"Result: {result}"
)