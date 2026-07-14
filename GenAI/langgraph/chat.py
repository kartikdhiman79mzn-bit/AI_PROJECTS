import os

from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable before running this script.")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class State(TypedDict):
    text: str
    
user_query = 'Hello How are you?'

def llm_call(state: State) -> dict:
    """ Call a LLM to response the user query """
    
    response = client.chat.completions.create(
        model='gemini-3.5-flash',
        messages=[
            { 'role': 'system', 'content': 'You are a helpful ai assistant.'},
            { 'role': 'user', 'content': user_query }
        ]
    )
    
    return { 'text': state['text'] + response.choices[0].message.content  }


# Graph Builder
graph_builder = StateGraph(State)

# Add node to a graph
graph_builder.add_node("node_a", llm_call)

# Add edges to node (Serial Edge)
graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", END)

#graph = graph_builder.compile()

print(graph_builder.compile().get_graph().draw_mermaid())

#print(graph.invoke(State({ 'text': '' })))