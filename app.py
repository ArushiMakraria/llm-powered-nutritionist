import chainlit as cl
import utils
from langchain_core.messages import HumanMessage
import uuid
import os

# Define global variables for workflow and config
workflow = None
config = None

@cl.on_chat_start
async def init_graph():
    """
    Initializes the workflow graph and configuration when the chat starts.
    """
    global workflow, config
    try:
        # Create the workflow using utils
        workflow = utils.create_graph()
        # Generate a unique thread ID for this session
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id": thread_id}}

        await cl.Message(content="""Hey there, welcome to Nutrisense AI, where we take care of your daily dietary needs! 
                         Don't know what to make today? Worry not, we got you covered. Just tell me what you have and we'll do the rest.""").send()
    except Exception as e:
        # Handle any exceptions and send the error message to the UI
        await cl.Message(content=f"Error: {str(e)}").send()


@cl.on_message
async def query(message: cl.Message):
    """
    Processes user input and streams responses from the workflow graph.
    """
    global workflow, config
    try:
        if workflow is None or config is None:
            raise ValueError("Workflow or config not initialized. Please restart the session.")

        # Create input for the workflow
        inputs = {"messages": [HumanMessage(content=message.content)]}
        # print("Inputs: ", inputs)
        # Stream messages from the workflow
        async for s in workflow.astream(inputs, stream_mode="values", config=config):

            if "messages" in s:
                last_message = s["messages"][-1]
                
                # Send messages to Chainlit UI
                if isinstance(last_message, str):
                    await cl.Message(content=last_message).send()
                else:
                    if last_message.name == "recipe_node" or last_message.name == "visual_node":
                        await cl.Message(content=last_message.content).send()
        # if os.path.exists("plot.png"):
        await cl.Message(content = "Here is the nutritional information graph:",elements=[cl.Image(path="plot.png", size="large")]).send()
    except Exception as e:
        # Handle any exceptions and send the error message to the UI
        await cl.Message(content=f"Error: {str(e)}").send()
