import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os
from main import create_nutritionist_workflow

# Store conversation context
conversation_context = []

@cl.on_chat_start
async def init_graph():
    """
    Initializes the workflow graph when the chat starts.
    """
    global conversation_context
    conversation_context = []  # Reset conversation context
    
    await cl.Message(content="""# 🍳 Welcome to Nutrisense AI!

**Your personal nutrition and recipe assistant**

## What I can help you with:
• 🥘 **Specific recipes** (e.g., "Palak Paneer", "Tahini Dressing")
• 🍽️ **Custom meal ideas** (e.g., "high-protein breakfast under 500 calories")
• 📅 **Multi-day meal plans** and meal prep guidance
• 🥗 **Nutritional information** about foods and nutrients
• 🔄 **Ingredient substitutions** and dietary accommodations

## How to get the best results:
- Be specific about what you want (recipe name, meal type, restrictions)
- Mention any ingredients you don't have or want to avoid
- Include your dietary preferences and nutritional goals

**Ready to start? Just tell me what you'd like to make or learn about!**""").send()


async def send_chunked_message(content: str, max_length: int = 2000):
    """
    Send long content in chunks to avoid payload overflow.
    """
    if len(content) <= max_length:
        await cl.Message(content=content).send()
        return
    
    # Split content into chunks
    chunks = []
    current_chunk = ""
    lines = content.split('\n')
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send chunks with small delays
    for i, chunk in enumerate(chunks):
        if i == 0:
            await cl.Message(content=chunk).send()
        else:
            await cl.Message(content=f"**Continued...**\n\n{chunk}").send()


@cl.on_message
async def query(message: cl.Message):
    """
    Processes user input and streams responses from the enhanced workflow graph.
    """
    global conversation_context
    try:
        print(f"🔍 DEBUG: Starting query processing for: {message.content}")
        
        # Create a fresh workflow for each request
        workflow = create_nutritionist_workflow()
        
        # Use a unique thread ID for each request
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Create input with the current user message
        inputs = {"messages": [HumanMessage(content=message.content)]}
        
        # Add a processing message
        processing_msg = cl.Message(content="🔄 Analyzing your request and preparing response...")
        await processing_msg.send()
        
        # Track what we've displayed for this specific request
        displayed_intent = False
        displayed_recipe = False
        displayed_diet_plan = False
        displayed_nutritional_info = False
        displayed_visualization = False
        displayed_blocked = False
        
        # Use a flag to control the loop instead of break/return
        clinical_blocked = False
        
        print(f"🔍 DEBUG: Starting workflow stream...")
        
        # Stream messages from the workflow
        async for chunk in workflow.astream(inputs, stream_mode="values", config=config):
            print(f"🔍 DEBUG: Received chunk with keys: {list(chunk.keys())}")
            
            # Handle clinical guardrail messages
            if "clinical_check" in chunk:
                print(f"🔍 DEBUG: Processing clinical_check")
                clinical_check = chunk["clinical_check"]
                if clinical_check and clinical_check.is_clinical:
                    # Remove processing message
                    await processing_msg.remove()
                    
                    # Display the clinical guardrail message
                    if "messages" in chunk and chunk["messages"]:
                        for msg in chunk["messages"]:
                            if hasattr(msg, 'name') and msg.name == "clinical_guardrail":
                                clinical_msg = cl.Message(content=msg.content)
                                await clinical_msg.send()
            
                    clinical_blocked = True
            
            # Handle clinical guardrail blocks FIRST
            if "blocked" in chunk and chunk["blocked"] and not displayed_blocked:
                print(f"🔍 DEBUG: Processing blocked content")
                # Remove processing message
                await processing_msg.remove()
                
                # Get the clinical guardrail message
                if "messages" in chunk and chunk["messages"]:
                    for msg in chunk["messages"]:
                        if hasattr(msg, 'name') and msg.name == "clinical_guardrail":
                            block_msg = cl.Message(content=msg.content)
                            await block_msg.send()
                            displayed_blocked = True
                            return  # Exit early - workflow should end here
                
                # Fallback if no specific message found
                if not displayed_blocked:
                    block_msg = cl.Message(content=f"⚠️ **Safety Notice**: {chunk['blocked']}")
                    await block_msg.send()
                    return
            
            # Only process other nodes if not blocked
            if not clinical_blocked:
                # Handle intent analysis (more detailed but concise)
                if "intent" in chunk and chunk["intent"] and not displayed_intent:
                    print(f"🔍 DEBUG: Processing intent")
                    intent_data = chunk["intent"]
                    
                    # More informative intent display
                    intent_content = f"## 🎯 Request Analysis\n\n"
                    intent_content += f"**Type:** {intent_data.primary_intent.replace('_', ' ').title()}\n"
                    intent_content += f"**Specificity:** {intent_data.recipe_specificity.replace('_', ' ').title()}\n"
                    
                    if intent_data.meal_type:
                        intent_content += f"**Meal:** {', '.join(intent_data.meal_type).title()}\n"
                    
                    if intent_data.dietary_restrictions:
                        intent_content += f"**Restrictions:** {', '.join(intent_data.dietary_restrictions)}\n"
                    
                    if intent_data.excluded_ingredients:
                        intent_content += f"**Avoiding:** {', '.join(intent_data.excluded_ingredients)}\n"
                    
                    if intent_data.nutritional_requirements:
                        intent_content += f"**Focus:** {intent_data.nutritional_requirements}\n"
                    
                    intent_msg = cl.Message(content=intent_content)
                    await intent_msg.send()
                    displayed_intent = True
                
                # Handle single recipe recommendations (improved formatting)
                if "recipe" in chunk and chunk["recipe"] and not displayed_recipe:
                    print(f"🔍 DEBUG: Processing recipe")
                    recipe_data = chunk["recipe"]
                    
                    if hasattr(recipe_data, 'name'):
                        # Enhanced recipe display with better formatting
                        recipe_header = f"# 🍳 {recipe_data.name}\n\n"
                        
                        # Recipe info bar
                        info_bar = f"⏱️ **Prep:** {recipe_data.prep_time} | "
                        info_bar += f"🔥 **Cook:** {recipe_data.cook_time} | "
                        info_bar += f"🍽️ **Serves:** {recipe_data.servings}\n\n"
                        
                        # Detailed nutrition
                        nutrition_section = f"## 📊 Nutritional Information\n"
                        nutrition_section += f"{recipe_data.nutritional_info}\n\n"
                        
                        # Send header and nutrition first
                        await cl.Message(content=recipe_header + info_bar + nutrition_section).send()
                        
                        # Send ingredients section
                        ingredients_section = f"## 🛒 Ingredients\n"
                        for ingredient in recipe_data.ingredients:
                            ingredients_section += f"• {ingredient}\n"
                        ingredients_section += "\n"
                        
                        await cl.Message(content=ingredients_section).send()
                        
                        # Send instructions section
                        instructions_section = f"## 👨‍🍳 Instructions\n"
                        for i, instruction in enumerate(recipe_data.instructions, 1):
                            instructions_section += f"**{i}.** {instruction}\n\n"
                        
                        await send_chunked_message(instructions_section)
                    else:
                        await cl.Message(content=f"🍳 **Recipe**:\n{str(recipe_data)[:1000]}...").send()
                    
                    displayed_recipe = True
                
                # Handle diet plan recommendations (ENHANCED to show full recipes)
                if "diet_plan" in chunk and chunk["diet_plan"] and not displayed_diet_plan:
                    print(f"🔍 DEBUG: Processing diet_plan")
                    diet_plan_data = chunk["diet_plan"]
                    
                    if hasattr(diet_plan_data, 'plan_name'):
                        # Send diet plan header
                        diet_header = f"# 📅 {diet_plan_data.plan_name}\n\n"
                        diet_header += f"**Duration:** {diet_plan_data.duration}\n"
                        if hasattr(diet_plan_data, 'total_nutritional_info'):
                            diet_header += f"**Nutritional Summary:** {diet_plan_data.total_nutritional_info}\n\n"
                        await cl.Message(content=diet_header).send()
                        
                        # Send each day's plan with FULL RECIPE DETAILS
                        if hasattr(diet_plan_data, 'daily_plans'):
                            print(f"🔍 DEBUG: Processing {len(diet_plan_data.daily_plans)} daily plans")
                            for day_plan in diet_plan_data.daily_plans:
                                day_content = f"## {day_plan.day}\n\n"
                                await cl.Message(content=day_content).send()
                                
                                # Process each meal type with FULL recipe details
                                meal_types = [
                                    ("🌅 Breakfast", day_plan.breakfast),
                                    ("🌞 Lunch", day_plan.lunch),
                                    ("🌙 Dinner", day_plan.dinner),
                                    ("🍎 Snacks", day_plan.snack)
                                ]
                                
                                for meal_emoji_name, recipes in meal_types:
                                    if recipes:
                                        meal_header = f"### {meal_emoji_name}\n\n"
                                        await cl.Message(content=meal_header).send()
                                        
                                        for recipe in recipes:
                                            # Full recipe details for each recipe
                                            recipe_content = f"#### 🍳 {recipe.name}\n\n"
                                            recipe_content += f"⏱️ **Prep:** {recipe.prep_time} | 🔥 **Cook:** {recipe.cook_time} | 🍽️ **Serves:** {recipe.servings}\n"
                                            recipe_content += f"📊 **Nutrition:** {recipe.nutritional_info}\n\n"
                                            
                                            # Add ingredients
                                            recipe_content += f"**🛒 Ingredients:**\n"
                                            for ingredient in recipe.ingredients:
                                                recipe_content += f"• {ingredient}\n"
                                            recipe_content += "\n"
                                            
                                            # Add instructions
                                            recipe_content += f"**👨‍🍳 Instructions:**\n"
                                            for i, instruction in enumerate(recipe.instructions, 1):
                                                recipe_content += f"{i}. {instruction}\n"
                                            recipe_content += "\n"
                                            
                                            await send_chunked_message(recipe_content)
                        
                        # Send shopping list as final section
                        if hasattr(diet_plan_data, 'shopping_list') and diet_plan_data.shopping_list:
                            shopping_content = f"## 🛒 Complete Shopping List\n\n"
                            
                            # Group items for better organization
                            for i, item in enumerate(diet_plan_data.shopping_list[:25], 1):
                                shopping_content += f"{i}. {item}\n"
                            
                            if len(diet_plan_data.shopping_list) > 25:
                                shopping_content += f"\n*...and {len(diet_plan_data.shopping_list) - 25} more items*\n"
                            
                            await send_chunked_message(shopping_content)
                    else:
                        await cl.Message(content=f"📅 **Diet Plan**:\n{str(diet_plan_data)[:1000]}...").send()
                    
                    displayed_diet_plan = True
                
                # Handle nutritional information (better organization)
                if "nutritional_info" in chunk and chunk["nutritional_info"] and not displayed_nutritional_info:
                    print(f"🔍 DEBUG: Processing nutritional_info")
                    nutritional_data = chunk["nutritional_info"]
                    
                    if hasattr(nutritional_data, 'query_summary'):
                        # Send header
                        nutrition_header = f"# 🥗 Nutritional Information\n\n"
                        nutrition_header += f"**Your Question:** {nutritional_data.query_summary}\n\n"
                        await cl.Message(content=nutrition_header).send()
                        
                        # Send food recommendations
                        if nutritional_data.food_recommendations:
                            food_content = f"## 🍎 Top Food Recommendations\n\n"
                            for i, food in enumerate(nutritional_data.food_recommendations[:12], 1):
                                food_content += f"{i}. **{food}**\n"
                            
                            if len(nutritional_data.food_recommendations) > 12:
                                food_content += f"\n*...and {len(nutritional_data.food_recommendations) - 12} more options*\n"
                            
                            await send_chunked_message(food_content)
                        
                        # Send nutritional breakdown
                        if nutritional_data.nutritional_breakdown:
                            breakdown_content = f"## 📊 Nutritional Details\n\n"
                            for nutrient, info in nutritional_data.nutritional_breakdown.items():
                                breakdown_content += f"**{nutrient.title()}:** {info}\n\n"
                            
                            await send_chunked_message(breakdown_content)
                        
                        # Send food sources by category
                        if nutritional_data.food_sources:
                            sources_content = f"## 🌱 Food Sources by Category\n\n"
                            for category, foods in nutritional_data.food_sources.items():
                                sources_content += f"**{category.title()}:** {', '.join(foods[:8])}"
                                if len(foods) > 8:
                                    sources_content += f" *(+{len(foods)-8} more)*"
                                sources_content += "\n\n"
                            
                            await send_chunked_message(sources_content)
                        
                        # Send additional notes
                        if nutritional_data.additional_notes:
                            notes_content = f"## 💡 Additional Tips\n\n{nutritional_data.additional_notes}"
                            await send_chunked_message(notes_content)
                    else:
                        await cl.Message(content=f"🥗 **Nutritional Information**:\n{str(nutritional_data)[:1000]}...").send()
                    
                    displayed_nutritional_info = True
                
                # Handle visualization
                if "visualization" in chunk and chunk["visualization"] and not displayed_visualization:
                    print(f"🔍 DEBUG: Processing visualization")
                    plot_path = chunk["visualization"]
                    if os.path.exists(plot_path):
                        viz_msg = cl.Message(
                            content="## 📈 Nutritional Visualization",
                            elements=[cl.Image(path=plot_path, name="nutrition_chart", display="inline")]
                        )
                        await viz_msg.send()
                    else:
                        await cl.Message(content=f"📊 Visualization saved to: {plot_path}").send()
                    displayed_visualization = True
        
        print(f"🔍 DEBUG: Workflow stream completed")
        
        # Remove processing message if still there and not clinical
        if not clinical_blocked:
            await processing_msg.remove()
        
        # Keep conversation context manageable (last 6 entries)
        if len(conversation_context) > 6:
            conversation_context = conversation_context[-6:]
            
    except Exception as e:
        print(f"❌ ERROR in app.py: {e}")
        print(f"🔍 DEBUG: Exception type: {type(e)}")
        # Handle any exceptions and send the error message to the UI
        error_msg = cl.Message(content=f"❌ **Error**: {str(e)}")
        await error_msg.send()