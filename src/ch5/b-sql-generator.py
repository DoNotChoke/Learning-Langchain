from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_together.chat_models import ChatTogether

from dotenv import load_dotenv
import os

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

model_low_temp = ChatTogether(api_key=TOGETHER_API_KEY, temperature=0)
model_high_temp = ChatTogether(api_key=TOGETHER_API_KEY, temperature=0.7)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    sql_query: str
    sql_explanation: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explanation: str

generate_prompt = SystemMessage(
    "You are a helpful data analyst, who generates SQL queries for users based on their questions."
)

def generate_state(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)

    return {
        "sql_query": res.content,
        "messages": [user_message, res]
    }

explain_prompt = SystemMessage(
    "You are a helpful data analyst, who explains SQL queries to users."
)

def explain_state(state: State) -> State:
    messages = [
        explain_prompt,

        *state["messages"],
    ]
    res = model_high_temp.invoke(messages)
    return {
        "sql_explanation": res.content,
        "messages": res
    }

builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("generate_state", generate_state)
builder.add_node("explain_state", explain_state)
builder.add_edge(START, "generate_state")
builder.add_edge("generate_state", "explain_state")
builder.add_edge("explain_state", END)

graph = builder.compile()

result = graph.invoke({"user_query": "What is the total sales for each product?"})
print(result)