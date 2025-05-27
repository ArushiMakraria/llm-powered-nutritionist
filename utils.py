from dotenv import load_dotenv
import matplotlib.pyplot as plt
import operator
import pandas as pd
import seaborn as sns
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI
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
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0,
    )

def get_search_tool():
    return DuckDuckGoSearchRun()

def get_python_tool():
    return PythonAstREPLTool()

def get_react_agent(tools: list):
    return create_react_agent(get_llm(), tools, )

