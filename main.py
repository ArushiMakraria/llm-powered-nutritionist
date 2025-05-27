from typing import Dict, List, Tuple, Any
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from intent import intent_node
from recipe import recipe_node
from diet_plan import diet_plan_node
from nutritional_info import nutritional_info_node
from visualization import visual_node
from grocery import grocery_node
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from guardrail import clinical_guardrail_node
from state import NutritionistState
from fallback import fallback_node
import json
from datetime import datetime

def pretty_print_chunk(chunk: Dict[str, Any]) -> None:
    """
    Pretty print a chunk from the LangGraph stream with proper formatting.
    
    Args:
        chunk: The chunk data from the workflow stream
    """
    print("\n" + "="*60)
    print(f"🔄 WORKFLOW UPDATE - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    for node_name, node_data in chunk.items():
        print(f"\n📍 Node: {node_name.upper()}")
        print("-" * 40)
        
        if isinstance(node_data, dict):
            for key, value in node_data.items():
                print(f"  {key}: ", end="")
                
                # Handle different types of values
                if isinstance(value, str):
                    # Truncate long strings for readability
                    if len(value) > 100:
                        print(f"{value[:100]}...")
                    else:
                        print(value)
                elif isinstance(value, list):
                    print(f"[{len(value)} items]")
                    for i, item in enumerate(value[:3]):  # Show first 3 items
                        print(f"    {i+1}. {str(item)[:50]}{'...' if len(str(item)) > 50 else ''}")
                    if len(value) > 3:
                        print(f"    ... and {len(value) - 3} more items")
                elif isinstance(value, dict):
                    print("📋 Dictionary:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {str(sub_value)[:50]}{'...' if len(str(sub_value)) > 50 else ''}")
                else:
                    print(str(value))
        else:
            print(f"  Data: {str(node_data)[:100]}{'...' if len(str(node_data)) > 100 else ''}")
    
    print("\n" + "="*60)

def create_nutritionist_workflow():
    """Creates the enhanced nutritionist workflow with conditional routing."""
    # Create memory for persistence
    memory = MemorySaver()
    
    # Create the graph
    graph = StateGraph(NutritionistState)
    
    # Add all nodes
    graph.add_node("clinical_guardrail", clinical_guardrail_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("recipe_node", recipe_node)
    graph.add_node("diet_plan_node", diet_plan_node)
    graph.add_node("nutritional_info_node", nutritional_info_node)
    graph.add_node("visualization_node", visual_node)
    graph.add_node("grocery_node", grocery_node)
    
    # Define conditional logic for clinical guardrail
    def should_continue_after_guardrail(state: NutritionistState):
        """Determine if workflow should continue or end based on clinical check."""
        print(f"🔍 DEBUG: Checking clinical guardrail...")
        clinical_check = state.get("clinical_check")
        if clinical_check and clinical_check.is_clinical:
            print(f"🚫 DEBUG: Clinical query detected, ending workflow")
            return "__end__"
        print(f"✅ DEBUG: Non-clinical query, proceeding to intent analysis")
        return "intent_node"
    
    # Define conditional logic for intent routing
    def route_after_intent(state: NutritionistState):
        """Route to appropriate node based on intent classification."""
        print(f"🔍 DEBUG: Routing after intent analysis...")
        intent = state.get("intent")
        if not intent:
            print(f"⚠️ DEBUG: No intent found, defaulting to recipe_node")
            return "recipe_node"  # Default fallback
        
        print(f"🎯 DEBUG: Intent primary_intent = {intent.primary_intent}")
        
        if intent.primary_intent == "nutritional_info":
            print(f"📊 DEBUG: Routing to nutritional_info_node")
            return "nutritional_info_node"
        elif intent.primary_intent == "diet_plan":
            print(f"📅 DEBUG: Routing to diet_plan_node")
            return "diet_plan_node"
        else:  # single_recipe, ingredient_substitution, meal_prep
            print(f"🍳 DEBUG: Routing to recipe_node")
            return "recipe_node"
    
    # Define conditional logic after content generation
    def route_to_visualization(state: NutritionistState):
        """Route to visualization based on what content was generated."""
        print(f"🔍 DEBUG: Checking for visualization routing...")
        print(f"🔍 DEBUG: State keys: {list(state.keys())}")
        
        if state.get("nutritional_info"):
            print(f"📊 DEBUG: Found nutritional_info, routing to visualization")
            return "visualization_node"
        elif state.get("diet_plan"):
            print(f"📅 DEBUG: Found diet_plan, routing to visualization")
            return "visualization_node"
        elif state.get("recipe"):
            print(f"🍳 DEBUG: Found recipe, routing to visualization")
            return "visualization_node"
        else:
            print(f"❌ DEBUG: No content found, ending workflow")
            return "__end__"
    
    # Add conditional edges
    graph.add_conditional_edges(
        "clinical_guardrail",
        should_continue_after_guardrail,
        {
            "intent_node": "intent_node",
            "__end__": END
        }
    )
    
    # Route from intent to appropriate content generation node
    graph.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "recipe_node": "recipe_node",
            "diet_plan_node": "diet_plan_node",
            "nutritional_info_node": "nutritional_info_node"
        }
    )
    
    # Route from content generation to visualization
    graph.add_conditional_edges(
        "recipe_node",
        route_to_visualization,
        {
            "visualization_node": "visualization_node",
            "__end__": END
        }
    )
    
    graph.add_conditional_edges(
        "diet_plan_node",
        route_to_visualization,
        {
            "visualization_node": "visualization_node",
            "__end__": END
        }
    )
    
    graph.add_conditional_edges(
        "nutritional_info_node",
        route_to_visualization,
        {
            "visualization_node": "visualization_node",
            "__end__": END
        }
    )
    
    # End after visualization
    graph.add_edge("visualization_node", END)
    
    # Set entry point to clinical guardrail
    graph.set_entry_point("clinical_guardrail")
    
    # Compile the graph
    workflow = graph.compile(checkpointer=memory)
    
    return workflow

def process_user_query(query: str) -> Dict[str, Any]:
    """
    Process a user's nutrition query through the enhanced workflow.
    
    Args:
        query: The user's nutrition-related question
    
    Returns:
        Dict containing the workflow results including intent analysis,
        recipe recommendations, diet plans, nutritional info, and visualizations
    """
    # Create the workflow
    workflow = create_nutritionist_workflow()
    
    # Create the initial state
    state = NutritionistState(messages=[HumanMessage(content=query)])
    
    # Run the workflow with pretty printing
    thread_id = "1"
    print(f"\n🚀 Starting Enhanced Nutritionist Workflow")
    print(f"📝 Query: {query}")
    print(f"🔗 Thread ID: {thread_id}")
    
    final_result = None
    
    for chunk in workflow.stream(
        state, 
        stream_mode="values",  # Use "values" to get full state after each step
        config={"configurable": {"thread_id": thread_id}}
    ):
        print(chunk)
        print("--------------------------------")
        final_result = chunk
    
    print(f"\n✅ Workflow completed successfully!")
    return final_result

if __name__ == "__main__":
    # Test cases for different scenarios
    test_queries = [
        "I need a high-protein breakfast recipe that's gluten-free and under 500 calories",
        "I need a week of meal prep ideas for lunch and dinner",
        "What foods are high in calcium but don't contain dairy?",
        "How to make palak paneer",
        "I don't have eggs, what can I make for breakfast?",
        "How to make tahini dressing"
    ]
    
    # Test with the first query
    query = test_queries[0]
    result = process_user_query(query)
    
    # Print final summary
    print("\n" + "🎯 FINAL RESULTS SUMMARY")
    print("="*60)
    if result:
        for key, value in result.items():
            if key != "messages":  # Skip messages for cleaner output
                print(f"{key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
