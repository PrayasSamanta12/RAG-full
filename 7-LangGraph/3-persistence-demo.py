import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# --- 1. Define State ---
# We use 'add_messages' to tell LangGraph to APPEND new messages to the history
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define a simple model (using OpenAI or a mock if no key)
# If you don't have a key set, this part might fail, so we'll add a fallback
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
except:
    print("OpenAI Key not found, using mock response.")
    llm = None

def chatbot(state: State):
    messages = state["messages"]
    
    if llm:
        response = llm.invoke(messages)
        return {"messages": [response]}
    else:
        # Mock response for demonstration without API key
        last_user_msg = messages[-1].content
        return {"messages": [f"Echo: {last_user_msg} (History length: {len(messages)})"]}

# --- 2. Build Graph with Checkpointer ---
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Initialize the checkpointer (In-Memory for this demo)
# For permanent storage, use: from langgraph.checkpoint.sqlite import SqliteSaver
memory = MemorySaver()

# COMPILE WITH CHECKPOINTER
graph = builder.compile(checkpointer=memory)

# --- 3. Run with Thread ID ---
# This config is the key. 'thread_id' determines WHICH history stack to load.
config = {"configurable": {"thread_id": "session_A"}}

print("--- Turn 1 (Session A) ---")
input_1 = {"messages": [HumanMessage(content="Hi, I am Alice.")]}
resp_1 = graph.invoke(input_1, config=config)
print(f"Bot: {resp_1['messages'][-1].content}")

print("\n--- Turn 2 (Session A) ---")
# functionality: LangGraph loads the history for 'session_A' automatically.
# We DO NOT pass "Hi, I am Alice" again.
input_2 = {"messages": [HumanMessage(content="What is my name?")]}
resp_2 = graph.invoke(input_2, config=config)
print(f"Bot: {resp_2['messages'][-1].content}")

print("\n--- Turn 1 (Session B) ---")
# New thread_id means new history (clean slate)
config_b = {"configurable": {"thread_id": "session_B"}}
input_3 = {"messages": [HumanMessage(content="Hi, I am Bob.")]}
resp_3 = graph.invoke(input_3, config_b)
print(f"Bot: {resp_3['messages'][-1].content}")
