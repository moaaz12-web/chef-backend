from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from typing import List, Dict, Annotated
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.store.base import BaseStore
from langgraph.checkpoint.memory import MemorySaver
import uuid
from typing_extensions import TypedDict
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")

# Initialize model and MongoDB client
model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
from langgraph.checkpoint.mongodb import MongoDBSaver

mongodb_client = MongoClient(mongodb_uri)

from langgraph.checkpoint.mongodb import MongoDBSaver
checkpointer = MongoDBSaver(mongodb_client)


from langchain_core.messages import SystemMessage
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


def generate_system_message(preferences: Dict[str, str]):
    base_msg = """You are a knowledgeable and friendly chef and health assistant. Your role is to strictly answer questions and provide information ONLY about cuisines, recipes, cooking techniques, and general health advice related to food and nutrition, based on the user's preferences.
    If a user asks about a topic that is unrelated to cooking or health, politely inform them that you can only assist with food and health-related queries. Never make assumptions—always ask the user for clarification if their query is unclear or you need more details to provide an accurate response.
    """

    prefs_msg = " ".join([f"{key}: {value}." for key, value in preferences.items()])
    return f"{base_msg} User preferences: {prefs_msg}"


def call_model(state: MessagesState, config: RunnableConfig):
    preferences = config['configurable']['preferences']
    system_msg = generate_system_message(preferences)

    print(state['messages'][-1].content[0])

    response = model.invoke(
        [{"role": "user", "content": system_msg}] + state['messages']
    )
    return {"messages": response} 

# Build and compile the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

# FastAPI application setup
app = FastAPI()

# Define the input model for the API
class MessageRequest(BaseModel):
    content: List[dict]
    config: Dict

@app.post("/ask")
async def ask_model(request: MessageRequest):
    try:
        content = request.content
        config = request.config

        # Create HumanMessage from content
        message = HumanMessage(content=content)
        
        # else:
        res = graph.invoke({"messages": [message]}, config)
        response = res['messages'][-1].content
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)


# somehow we can save previous 5 msgs or summary to reduce costs
# or use postgres?
