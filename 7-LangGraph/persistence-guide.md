# How to Store History in LangGraph

In LangGraph, history is **not** stored inside the graph object itself. It is managed by an external **Checkpointer** that saves the state at every step.

## The 3-Step Procedure

To make your graph remember conversation history, follow these three steps:

### 1. Define State with `add_messages`
You must use `Annotated` with `add_messages` in your state. This tells LangGraph to **append** new messages to the existing list instead of overwriting them.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 'add_messages' is CRITICAL. It ensures history is preserved.
    messages: Annotated[list, add_messages]
```

### 2. Compile with a Checkpointer
When you compile the graph, you must provide a `checkpointer`. This component handles the saving to memory or disk.

```python
from langgraph.checkpoint.memory import MemorySaver 
# For persistence (files/DB), use SqliteSaver or PostgresSaver

memory = MemorySaver() # Initialize the storage

# Attach storage to the graph
graph = builder.compile(checkpointer=memory) 
```

### 3. Invoke with a `thread_id`
When you run the graph, you **must** provide a `thread_id` in the config.

```python
# The unique ID for this user's conversation
config = {"configurable": {"thread_id": "user_session_1"}}

# Turn 1
graph.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

# Turn 2 - LangGraph loads the history automatically using 'thread_id'
graph.invoke({"messages": [HumanMessage(content="What did I just say?")]}, config=config)
```

## Storage Options

| Option | Description | Use Case |
| :--- | :--- | :--- |
| **MemorySaver** | Stores in RAM. Lost if script restarts. | Testing / Demos |
| **SqliteSaver** | Stores in a local `.sqlite` file. | Local Development |
| **PostgresSaver** | Stores in a Postgres database. | Production |
