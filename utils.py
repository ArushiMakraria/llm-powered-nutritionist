from dotenv import load_dotenv
import matplotlib.pyplot as plt
import operator
import pandas as pd
import seaborn as sns
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langgraph.graph import MessagesState, StateGraph, add_messages, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing import Sequence, Literal, TypedDict
from typing_extensions import Annotated
import config
from langgraph.checkpoint.memory import MemorySaver
import uuid
load_dotenv()

def get_llm():
    return ChatVertexAI(model_name="gemini-2.0-flash-exp", seed=42, temperature=0, streaming=True)

def get_search_tool():
    return DuckDuckGoSearchRun()

def get_python_tool():
    return PythonAstREPLTool()

def get_react_agent(tools: list):
    return create_react_agent(get_llm(), tools, )

def create_visual_agent():
    visualize_prompt = ChatPromptTemplate(
                                        [
                                            ("system", config.AGENT_PROMPT_VISUALIZATION),
                                            ("placeholder", "{messages}"),
                                        ]
                                        )
    visual_agent = create_react_agent(
                                        model=get_llm(),
                                        tools=[PythonAstREPLTool()],
                                        state_modifier=visualize_prompt,
                                        # state_schema=AgentState
                                    )
    return visual_agent

def visual_node(state: MessagesState) -> Command:
    visual_agent = create_visual_agent()
    result = visual_agent.invoke(state)
    return Command(
                    update={
                        "messages": [
                            HumanMessage(content=result["messages"][-1].content, name="visual_node")
                        ]
                        },
                        # goto="supervisor",
                    )

def create_recipe_agent():
    df = pd.read_csv("clean-food.csv")
    recipe_prompt = ChatPromptTemplate(
                                        [
                                            ("system", config.AGENT_PROMPT_RECIPE),
                                            ("placeholder", "{messages}"),
                                        ],
                                        partial_variables={"df_str": df.to_markdown(index=True)},  # Include the dataframe as markdown
                                    )
    recipe_agent = create_react_agent(
                                    model=get_llm(),
                                    tools=[DuckDuckGoSearchRun()],
                                    state_modifier=recipe_prompt,
                                    # state_schema=AgentState
                                )
    return recipe_agent

def recipe_node(state: MessagesState) -> Command:
    recipe_agent = create_recipe_agent()
    result = recipe_agent.invoke(state)
    return Command(
                    update={
                        "messages": [
                            HumanMessage(content=result["messages"][-1].content, name="recipe_node")
                        ]
                        },
                        # goto="visual_agent",
                    )

def create_graph():
    memory = MemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("recipe_node", recipe_node)
    graph.add_node("visual_node", visual_node)
    # graph.add_edge(START, "recipe_node")
    graph.add_edge("recipe_node", "visual_node")
    # graph.add_edge("visual_node", END)
    # graph.add_edge("visual_node", END)

    graph.set_entry_point("recipe_node")
    graph.set_finish_point("visual_node")

    workflow = graph.compile(checkpointer=memory)

    return workflow

