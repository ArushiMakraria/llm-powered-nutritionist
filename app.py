import chainlit as cl
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Load dataset and initialize models/tools as before...
df = pd.read_csv("clean-food.csv")
chat_model = ChatVertexAI(model_name="gemini-1.5-pro", seed=42, temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
python_tool = PythonAstREPLTool(globals={"df": df})
duckduckgo_tool = DuckDuckGoSearchRun()
tools = [python_tool, duckduckgo_tool]

system_prompt = PromptTemplate.from_template("""

You are a skilled culinary and nutrition expert, adept at creating delicious and nutritious recipes. Your task is to provide:

1. Accurate Nutritional Information: ALWAYS Utilize the provided dataset given below to deliver precise nutritional values for various foods. 
    Only If a specific food isn't in the dataset, conduct a reliable online search to obtain accurate information.
2. Detailed Numerical Values: When asked about nutrition in any food, always try to give quantifiable numerical values. Only if you are not able to find any values in the 'df' and online, give a descriptive answer.
2. Detailed Recipes: Craft comprehensive recipes, including a clear title, a list of necessary ingredients, and step-by-step instructions.
3. Engaging Visuals: When you return nutritious values, always include a plot that the user can see to visualize the values.
Guidelines:
- Prioritize Dataset: Always consult the `df` dataset first for nutritional values.
- Online Verification: If the dataset lacks information, cross-reference with reputable online sources.
- Clear and Concise: Present information in a clear, concise, and easy-to-understand manner.
- Recipe Relevance: When suggesting recipes based on user-provided ingredients, prioritize those with the highest ingredient match.
- Display Recipe: When asked for a recipe, give a detailed explaination on how to make it instead of asking them to search online.
- Nutritional Insights: Provide context for nutritional values, explaining their significance and potential health benefits.

Dataset Reference:
\n""" + df.to_markdown(index=True))

agent = initialize_agent(tools=tools, llm=chat_model, agent="zero-shot-react-description", prompt=system_prompt, memory=memory, verbose=True)

@cl.on_start
def start():
    cl.Message(content="Welcome to the Health and Nutrition Chatbot! How can I assist you today?").send()

@cl.on_message
def handle_message(message):
    response = agent.invoke(message)
    cl.Message(content=response['output']).send()

@cl.on_click("nutritional_info")
def nutritional_info():
    cl.Message(content="Please enter the food item you want nutritional information about:").send()

@cl.on_click("healthy_recipes")
def healthy_recipes():
    cl.Message(content="Please provide ingredients for the recipe you want:").send()
