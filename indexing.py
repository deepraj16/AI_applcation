import os 
from langchain_mistralai.chat_models import ChatMistralAI

def setup_llm(): 
    llm = ChatMistralAI(api_key="lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc")
    return llm


