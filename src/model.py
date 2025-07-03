from langchain_together.chat_models import ChatTogether
from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_model_together():
    return ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=os.getenv("TOGETHER_API_KEY"))

def get_model_openai():
    return ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))