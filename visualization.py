from langgraph.types import Command
from langchain_core.messages import AIMessage
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import Dict, Tuple, Any, List
from state import NutritionistState

def parse_nutritional_info(nutritional_info: str) -> Dict[str, Tuple[float, str]]:
    """
    Parse nutritional information string into a dictionary with values and units.
    Handles various formats including "Approximately X", "X kcal", etc.
    
    Args:
        nutritional_info: String like "Calories: Approximately 476 calories, Protein: 38.15g, ..."
    
    Returns:
        Dict mapping nutrient names to (value, unit) tuples
    """
    nutrition_data = {}
    
    # Split by commas and process each nutrient
    nutrients = nutritional_info.split(',')
    
    for nutrient in nutrients:
        nutrient = nutrient.strip()
        if ':' in nutrient:
            # Split on colon to get name and value+unit
            name, value_unit = nutrient.split(':', 1)
            name = name.strip()
            value_unit = value_unit.strip()
            
            # Clean up the name (remove extra words)
            name = name.replace('Approximately', '').strip()
            
            # More flexible regex to handle various formats
            # Matches: "Approximately 476 calories", "38.15g", "476 kcal", etc.
            patterns = [
                r'(?:approximately\s+)?([\d.]+)\s*([a-zA-Z]+)',  # "Approximately 476 calories"
                r'([\d.]+)\s*([a-zA-Z]+)',  # "38.15g"
                r'([\d.]+)',  # Just numbers (assume grams)
            ]
            
            match = None
            for pattern in patterns:
                match = re.search(pattern, value_unit.lower())
                if match:
                    break
            
            if match:
                value = float(match.group(1))
                unit = match.group(2) if len(match.groups()) > 1 else 'g'
                
                # Standardize unit names
                unit_mapping = {
                    'calories': 'kcal',
                    'calorie': 'kcal',
                    'kcal': 'kcal',
                    'grams': 'g',
                    'gram': 'g',
                    'g': 'g',
                    'mg': 'mg',
                    'milligrams': 'mg',
                    'milligram': 'mg'
                }
                
                unit = unit_mapping.get(unit.lower(), unit)
                nutrition_data[name] = (value, unit)
    
    return nutrition_data

def create_nutrition_plot(nutritional_info: str, title: str = "Nutritional Information") -> str:
    """
    Creates a nutrition facts visualization plot from a nutritional info string.
    
    Args:
        nutritional_info: String containing nutritional information
        title: Title for the plot
    
    Returns:
        str: Path to the saved plot
    """
    print(f"🔍 DEBUG: Parsing nutritional info: {nutritional_info}")
    
    # Parse the nutritional information
    nutrition_data = parse_nutritional_info(nutritional_info)
    
    print(f"🔍 DEBUG: Parsed nutrition data: {nutrition_data}")
    
    if not nutrition_data:
        # If parsing fails, create a simple text-based visualization
        print(f"⚠️ DEBUG: No parsed data, creating text visualization")
        return create_text_visualization(nutritional_info, title)
    
    # Prepare data for plotting
    nutrients = []
    values = []
    units = []
    
    for nutrient, (value, unit) in nutrition_data.items():
        nutrients.append(nutrient.title())
        values.append(value)
        units.append(unit)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Nutrient': nutrients,
        'Value': values,
        'Unit': units
    })
    
    print(f"🔍 DEBUG: DataFrame created with {len(df)} nutrients")
    
    # Fix matplotlib threading issue by using Agg backend
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Create bar plot
    ax = sns.barplot(x='Nutrient', y='Value', data=df, hue='Nutrient', palette='deep', legend=False)
    
    # Customize the plot
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Nutrient', fontsize=12)
    plt.ylabel('Amount', fontsize=12)
    
    # Add value annotations with appropriate units
    for i, (v, unit) in enumerate(zip(df['Value'], df['Unit'])):
        ax.text(i, v + max(df['Value']) * 0.01, f'{v:.1f} {unit}', 
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with static path
    plot_path = 'plot.png'  # Simplified path
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ DEBUG: Plot saved to {plot_path}")
    return plot_path

def create_text_visualization(nutritional_info: str, title: str) -> str:
    """
    Create a simple text-based visualization when parsing fails.
    """
    import matplotlib
    matplotlib.use('Agg')
    
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"{title}\n\n{nutritional_info}", 
             ha='center', va='center', fontsize=12, wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    
    plot_path = 'plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def visual_node(state: NutritionistState) -> Dict[str, Any]:
    """
    Process visualization request from state without using LLM.
    Handles different content types (recipe, diet plan, nutritional info).
    Returns just the plot path for frontend display.
    """
    print(f"🔍 DEBUG: Starting visual_node...")
    print(f"🔍 DEBUG: State keys in visual_node: {list(state.keys())}")
    
    nutritional_info = None
    title = "Nutritional Information"
    
    try:
        # Check for different content types and extract nutritional info
        if state.get("recipe") and hasattr(state["recipe"], "nutritional_info"):
            print(f"🔍 DEBUG: Found recipe with nutritional info")
            nutritional_info = state["recipe"].nutritional_info
            if hasattr(state["recipe"], "name"):
                title = f"Nutritional Information - {state['recipe'].name}"
        
        elif state.get("diet_plan") and hasattr(state["diet_plan"], "total_nutritional_info"):
            print(f"🔍 DEBUG: Found diet plan with nutritional info")
            nutritional_info = state["diet_plan"].total_nutritional_info
            if hasattr(state["diet_plan"], "plan_name"):
                title = f"Nutritional Summary - {state['diet_plan'].plan_name}"
        
        elif state.get("nutritional_info"):
            print(f"🔍 DEBUG: Found nutritional_info directly")
            # Handle NutritionalInfo object
            if hasattr(state["nutritional_info"], "nutritional_breakdown"):
                # Convert nutritional breakdown dict to string format
                breakdown = state["nutritional_info"].nutritional_breakdown
                info_parts = []
                for nutrient, value in breakdown.items():
                    info_parts.append(f"{nutrient}: {value}")
                nutritional_info = ", ".join(info_parts)
            else:
                nutritional_info = str(state["nutritional_info"])
        
        if not nutritional_info:
            print(f"🔍 DEBUG: No nutritional info found in any expected location")
            return {
                "messages": [AIMessage(content="No nutritional information available to visualize", name="visual_node")],
                "error": "No nutritional data found"
            }
        
        print(f"🔍 DEBUG: Creating plot with nutritional_info: {nutritional_info[:100]}...")
        
        # Create visualization and get plot path
        plot_path = create_nutrition_plot(nutritional_info, title)
        
        print(f"✅ DEBUG: Plot created successfully at {plot_path}")
        
        return {
            "visualization": plot_path,  # Just return the plot path as a string
            "messages": [AIMessage(content=f"Created nutrition visualization: {plot_path}", name="visual_node")]
        }
    
    except Exception as e:
        print(f"❌ ERROR in visual_node: {e}")
        print(f"🔍 DEBUG: Exception type: {type(e)}")
        
        # Create a fallback text visualization
        try:
            if nutritional_info:
                plot_path = create_text_visualization(nutritional_info, title)
                return {
                    "visualization": plot_path,
                    "messages": [AIMessage(content=f"Created text-based nutrition visualization: {plot_path}", name="visual_node")]
                }
        except Exception as fallback_error:
            print(f"❌ ERROR in fallback visualization: {fallback_error}")
        
        return {
            "messages": [AIMessage(content=f"Error creating visualization: {str(e)}", name="visual_node")],
            "error": str(e)
        }

