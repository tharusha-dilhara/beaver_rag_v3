from flask import Flask, request, jsonify
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import time
from pymongo import MongoClient
from bson import ObjectId
import json
from flask_cors import CORS
from flask_mail import Mail, Message
from collections import defaultdict
from datetime import datetime
import bcrypt 
import traceback
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



# Initialize ChatNVIDIA client
client = ChatNVIDIA(
    model="qwen/qwen2.5-7b-instruct",
    api_key="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",
    temperature=0.2,
    top_p=0.7,
    max_tokens=500,
)

# MongoDB connection
DB_URI = "mongodb+srv://slbasha5555:0dtBv6tvnVrRRQGY@cluster0.ozxk2.mongodb.net/beaver_h1?retryWrites=true&w=majority"
mongo_client = MongoClient(DB_URI)
db = mongo_client.beaver_h1
bills_collection = db.bills

# New MongoDB collection for storing reports
reports_collection = db.recipe_reports_helper

# New MongoDB collection for storing generated reports
generated_reports_collection = db.recipe_generated_reports


# New MongoDB collection for health conditions
health_conditions_collection = db.health_conditions

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

user_data = {
    # Format: {userid: {"index": faiss_index, "documents": list}}
}

def fetch_stock_data(userid, limit=100):
    """
    Fetch stock data directly from MongoDB
    """
    try:
        # Convert string userid to ObjectId
        user_obj_id = ObjectId(userid)
        
        # Find all bills for the user
        bills = bills_collection.find({"userid": user_obj_id})
        
        # Extract items from all bills
        stock_items = []
        for bill in bills:
            for item in bill.get('items', []):
                stock_items.append({
                    "date": bill['date'],
                    "name": item['itemName'],
                    "quantity": item['quantity'],
                    "price": item['price']
                })
        print(f"Found {len(stock_items)} stock items for user {userid}")
        return stock_items
        
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return []

def format_stock_documents(stock_items):
    """
    Format the stock items into document strings for the RAG system.
    """
    documents = []
    for item in stock_items:
        try:
            date_obj = time.strptime(item['date'].split('T')[0], "%Y-%m-%d")
            month = time.strftime("%B", date_obj)
        except:
            month = "Unknown"
        doc = f"{month}: Inventory: {item['name']}: {item['quantity']}: ${item['price']}"
        documents.append(doc)
    return documents

def update_index(userid):
    """
    Fetch fresh data and rebuild the FAISS index for a specific user.
    """
    stock_items = fetch_stock_data(userid)
    if not stock_items:
        return False
    documents = format_stock_documents(stock_items)
    doc_embeddings = embedder.encode(documents)
    doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
    dimension = doc_embeddings.shape[1]
    user_data[userid] = {
        "index": faiss.IndexFlatL2(dimension),
        "documents": documents
    }
    if len(doc_embeddings) > 0:
        user_data[userid]["index"].add(doc_embeddings)
    return True

def retrieve(query, userid, recipes_type, top_k=3):
    """
    Retrieve the top_k most relevant documents using query and recipes_type.
    """
    # Combine the query with the recipes type to add context
    combined_query = f"{query} {recipes_type}"
    
    update_index(userid)
    if userid not in user_data or user_data[userid]["index"].ntotal == 0:
        if not update_index(userid):
            return ["No data available."]
    
    user_index = user_data[userid]["index"]
    user_docs = user_data[userid]["documents"]
    query_vec = embedder.encode([combined_query])
    query_vec = np.array(query_vec, dtype=np.float32)
    distances, indices = user_index.search(query_vec, min(top_k, user_index.ntotal))
    return [user_docs[idx] for idx in indices[0]]

@app.route('/itemrag', methods=['POST'])
def get_rag_answer():
    """
    Flask endpoint for handling RAG queries with user-specific data.
    Returns recipes as a JSON array and saves the result to the database.
    """
    data = request.get_json()
    if not data or 'query' not in data or 'userid' not in data or 'recipes_type' not in data:
        return jsonify({"error": "Please provide 'query', 'userid', and 'recipes_type' in the JSON body."}), 400

    query = data['query']
    userid = data['userid']
    recipes_type = data['recipes_type']
    answer = only_item_rag_query(query, userid, recipes_type)

    # Save the result to the database
    try:
        reports_collection.insert_one({
            "userid": userid,
            "type": "itemrag",
            "query": query,
            "recipes_type": recipes_type,
            "result": answer,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"Error saving itemrag result to database: {e}")

    return jsonify({"answer": answer})

def only_item_rag_query(query, userid, recipes_type):
    """
    Build the prompt to get recipe names in JSON array format with improved clarity.
    """
    retrieved_docs = retrieve(query, userid, recipes_type)
    if retrieved_docs == ["No data available."]:
        return ["No recipes can be suggested with current inventory"]

    prompt = f"""
You are a culinary expert specializing in Sri Lankan {recipes_type} recipes.
The available inventory items are: {', '.join(retrieved_docs)}.
Provide creative {recipes_type} recipes that can be made using these ingredients, and allow at most 1-2 additional ingredients.
Ensure that the recipe suggestions are appropriate for {recipes_type} (for example, {recipes_type} recipes might be lighter for breakfast or heartier for dinner).
Return ONLY the recipe names as a JSON-formatted array.
Example output: ["Recipe1", "Recipe2", "Recipe3"].
Ensure that the output is valid JSON.
"""
    print(f"Prompt: {prompt}")
    try:
        response = client.invoke([{"role": "user", "content": prompt}])
        recipes = response.content.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed_recipes = json.loads(recipes)
            if isinstance(parsed_recipes, list):
                return parsed_recipes
            return ["Formatting error in response: Not a list."]
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {json_err}")
            recipes = [x.strip().lstrip('0123456789.- ').strip('"') 
                        for x in recipes.strip('[]').split(",")]
            return [r for r in recipes if r]
    except Exception as e:
        return ["Error generating recipes: " + str(e)]

@app.route('/cooking_guidance', methods=['POST'])
def get_cooking_guidance():
    """
    Flask endpoint for handling cooking guidance queries.
    Returns a JSON object with the recipe and step-by-step cooking instructions.
    """
    data = request.get_json()
    if not data or 'query' not in data or 'userid' not in data or 'recipes_type' not in data:
        return jsonify({"error": "Please provide 'query', 'userid', and 'recipes_type' in the JSON body."}), 400

    query = data['query']
    userid = data['userid']
    recipes_type = data['recipes_type']
    guidance = generate_cooking_guidance(query, userid, recipes_type)
    return jsonify({"cooking_guidance": guidance})

def generate_cooking_guidance(query, userid, recipes_type):
    """
    Build the prompt to get a recipe with detailed, step-by-step cooking guidance.
    """
    retrieved_docs = retrieve(query, userid, recipes_type)
    if retrieved_docs == ["No data available."]:
        return {"error": "No recipes can be suggested with current inventory"}

    prompt = f"""
You are a culinary expert specializing in Sri Lankan {recipes_type} recipes.
The available inventory items are: {', '.join(retrieved_docs)}.
Based on these ingredients, suggest one creative {recipes_type} recipe that can be prepared by adding at most 1-2 extra ingredients.
Provide detailed, step-by-step cooking guidance including preparation, cooking, and serving instructions.
Return your answer as a JSON object with two keys:
- "recipe": the name of the recipe.
- "steps": an array of step-by-step instructions.
Example output:
{{
  "recipe": "",
  "steps": [
    "Step 1: ",
    "Step 2: ",
    "Step 3: ",
    "Step 4: "
  ]
}}
Ensure that your response is valid JSON.
"""
    print(f"Cooking Guidance Prompt: {prompt}")
    try:
        response = client.invoke([{"role": "user", "content": prompt}])
        guidance = response.content.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed_guidance = json.loads(guidance)
            if isinstance(parsed_guidance, dict) and "recipe" in parsed_guidance and "steps" in parsed_guidance:
                return parsed_guidance
            return {"error": "Formatting error: JSON object missing required keys."}
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error in cooking guidance: {json_err}")
            return {"error": "Could not parse JSON from the response."}
    except Exception as e:
        return {"error": "Error generating cooking guidance: " + str(e)}

@app.route('/enhanced_recipes', methods=['POST'])
def get_enhanced_recipes():
    """
    Flask endpoint for enhanced recipe suggestions.
    Returns a JSON object with recipe names, existing items, and new items required.
    Saves the result to the database.
    """
    data = request.get_json()
    if not data or 'query' not in data or 'userid' not in data or 'recipes_type' not in data:
        return jsonify({"error": "Please provide 'query', 'userid', and 'recipes_type' in the JSON body."}), 400

    query = data['query']
    userid = data['userid']
    recipes_type = data['recipes_type']
    enhanced_recipes = generate_enhanced_recipes(query, userid, recipes_type)

    # Save the result to the database
    try:
        reports_collection.insert_one({
            "userid": userid,
            "type": "enhanced_recipes",
            "query": query,
            "recipes_type": recipes_type,
            "result": enhanced_recipes,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"Error saving enhanced recipes result to database: {e}")

    return jsonify({"enhanced_recipes": enhanced_recipes})

def generate_enhanced_recipes(query, userid, recipes_type):
    """
    Build the prompt to get recipes with existing and new items required.
    """
    retrieved_docs = retrieve(query, userid, recipes_type)
    if retrieved_docs == ["No data available."]:
        return [{"error": "No recipes can be suggested with current inventory"}]

    prompt = f"""
You are a culinary expert specializing in Sri Lankan {recipes_type} recipes.
The available inventory items are: {', '.join(retrieved_docs)}.
Based on these ingredients, suggest creative {recipes_type} recipes that can be prepared by adding at least 1-2 extra ingredients.
For each recipe, provide:
- The name of the recipe.
- A list of existing items required from the inventory.
- A list of new items required to complete the recipe.
Return your answer as a JSON array where each element is an object with three keys:
- "recipe": the name of the recipe.
- "existing_items": an array of items from the inventory.
- "new_items": an array of new items required.
Example output:
[
  {{
    "recipe": "Recipe1",
    "existing_items": ["item1", "item2"],
    "new_items": ["item3"]
  }},
  {{
    "recipe": "Recipe2",
    "existing_items": ["item4", "item5","item6","item7"],
    "new_items": ["item6", "item7"]
  }},
  {{
    "recipe": "Recipe2",
    "existing_items": ["item4", "item5"],
    "new_items": ["item6", "item7"]
  }},
  {{
    "recipe": "Recipe2",
    "existing_items": ["item4", "item5"],
    "new_items": ["item6", "item7","item7"]
  }}
]
Ensure that your response is valid JSON.
"""
    print(f"Enhanced Recipes Prompt: {prompt}")
    try:
        response = client.invoke([{"role": "user", "content": prompt}])
        enhanced_recipes = response.content.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed_recipes = json.loads(enhanced_recipes)
            if isinstance(parsed_recipes, list):
                return parsed_recipes
            return [{"error": "Formatting error in response: Not a list."}]
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error in enhanced recipes: {json_err}")
            return [{"error": "Could not parse JSON from the response."}]
    except Exception as e:
        return [{"error": "Error generating enhanced recipes: " + str(e)}]

@app.route('/generate_report/<userid>', methods=['GET'])
def generate_report(userid):
    """
    Flask endpoint to generate a report for a specific user.
    Analyzes saved results using an LLM, saves the report to the database grouped by month, and returns it in the response.
    """
    try:
        # Fetch all reports for the given userid
        user_reports = list(reports_collection.find({"userid": userid}))
        if not user_reports:
            return jsonify({"error": "No reports found for the given user."}), 404

        # Prepare data for LLM analysis
        report_data = []
        for report in user_reports:
            report_data.append({
                "type": report["type"],
                "query": report["query"],
                "recipes_type": report["recipes_type"],
                "result": report["result"]
            })

        # Build the LLM prompt
        prompt = f"""
You are an AI assistant tasked with analyzing user data. The following is the data for user ID {userid}:
{json.dumps(report_data, indent=2)}

Analyze the data and provide a summary report. The report should include:
1. The most common types of queries (e.g., itemrag, enhanced_recipes).
2. Insights into the most frequently suggested recipes.
3. Any patterns or trends in the user's queries and results.
4. Suggestions for improving the user's experience.
5. Items to add to next month's shopping list.

Return the report in a clear and concise format.
"""

        # Invoke the LLM to analyze the data
        print(f"LLM Prompt: {prompt}")
        try:
            response = client.invoke([{"role": "user", "content": prompt}])
            llm_report = response.content.strip()
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            return jsonify({"error": "Failed to analyze data using LLM."}), 500

        # Get the current month name
        month_name = time.strftime("%B")

        # Save the generated report to the database grouped by month
        try:
            generated_reports_collection.insert_one({
                "userid": userid,
                "month": month_name,
                "report": llm_report,
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"Error saving generated report to database: {e}")
            return jsonify({"error": "Failed to save the generated report."}), 500

        return jsonify({"message": f"Report generated and saved successfully for {month_name}.", "llm_report": llm_report})
    except Exception as e:
        print(f"Error generating report for user {userid}: {e}")
        return jsonify({"error": "Could not generate report."}), 500

@app.route('/fetch_reports', methods=['POST'])
def fetch_reports():
    """
    Flask endpoint to fetch all reports for a specific user.
    Accepts 'userid' in the request body and returns all reports from the generated_reports collection.
    """
    data = request.get_json()
    if not data or 'userid' not in data:
        return jsonify({"error": "Please provide 'userid' in the JSON body."}), 400

    userid = data['userid']
    try:
        # Fetch all reports for the given userid
        user_reports = list(generated_reports_collection.find({"userid": userid}))
        if not user_reports:
            return jsonify({"error": "No reports found for the given user."}), 404

        # Convert ObjectId to string for JSON serialization
        for report in user_reports:
            report["_id"] = str(report["_id"])

        return jsonify({"reports": user_reports}), 200
    except Exception as e:
        print(f"Error fetching reports for user {userid}: {e}")
        return jsonify({"error": "Could not fetch reports."}), 500


#-----------------------------helth session ----------------------------------------------------

@app.route('/add_health_condition', methods=['POST'])
def add_health_condition():
    data = request.get_json()
    if not data or 'userid' not in data or 'condition_name' not in data:
        return jsonify({"error": "Missing required fields: userid, condition_name"}), 400

    try:
        health_conditions_collection.insert_one({
            "userid": data['userid'],
            "condition_name": data['condition_name'],
            "date_diagnosed": data.get('date_diagnosed', "Unknown"),
            "severity": data.get('severity', "Not specified")
        })
        return jsonify({"message": "Health condition added successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#-------------------------------------------------------------

@app.route('/health_suggestions', methods=['POST'])
def health_suggestions():
    """
    Flask endpoint to provide health suggestions based on user health conditions and stock items.
    """
    data = request.get_json()
    if not data or 'userid' not in data:
        return jsonify({"error": "Please provide 'userid' in the JSON body."}), 400

    userid = data['userid']

    try:
        health_conditions = db.users.find({"username": "test2"})
        health_conditions_list = [str(condition) for condition in health_conditions]

        stock_items = fetch_stock_data(userid)
        print(f"Fetched stock items for user {userid}: {stock_items}")
        print("------------------------------------------------\n\n")

        prompt = f"""
                    You are a health and nutrition expert. The user has the following health conditions: {health_conditions_list}.
                    The available stock items are: {stock_items}.
                    Based on these, suggest healthy meal plans or dietary recommendations that align with the user's health conditions.
                    Provide suggestions in JSON format with the following structure:
                    [
                    {{
                        "meal": "Meal Name",
                        "ingredients": ["item1", "item2"],
                        "health_benefits": "Explanation of how this meal benefits the user's health."
                    }}
                    ]
                    Ensure the response is valid JSON.
                    """
        print(f"Health Suggestions Prompt: {prompt}")

        response = client.invoke([{"role": "user", "content": prompt}])
        suggestions = response.content.strip().replace("```json", "").replace("```", "").strip()

        # Parse the response
        try:
            parsed_suggestions = json.loads(suggestions)
            if isinstance(parsed_suggestions, list):
                return jsonify({"health_suggestions": parsed_suggestions}), 200
            return jsonify({"error": "Formatting error in response: Not a list."}), 500
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error in health suggestions: {json_err}")
            return jsonify({"error": "Could not parse JSON from the response."}), 500



        return jsonify({"health_conditions": ", ".join(health_conditions_list)}), 200


        

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Health suggestions feature is under development."}), 200





@app.route('/test', methods=['GET'])
def test():

    stock = fetch_stock_data("650a1b2c3d4e5f6a7b8c9d0e")
    """
    Flask endpoint to test the server.
    Returns a simple message indicating the server is running.
    """
    return jsonify({"message": stock}), 200

@app.route('/update_current_status/<userid>', methods=['PUT'])
def update_current_status(userid):
    """
    Flask endpoint to update or add new fields to the current_status of a user.
    Accepts a JSON body with the fields to update or add.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Please provide data to update."}), 400

    try:
        # Fetch the user from the database
        user = db.users.find_one({"_id": ObjectId(userid)})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Update the current_status field
        current_status = user.get("current_status", {})
        current_status.update(data)  # Merge the new data into the existing current_status

        # Save the updated user data back to the database
        db.users.update_one({"_id": ObjectId(userid)}, {"$set": {"current_status": current_status}})
        return jsonify({"message": "Current status updated successfully", "current_status": current_status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#--------------------------------------------Essntial item list calculate------------------------------------------------------------

@app.route('/essential_items', methods=['POST'])
def essential_items():
    data = request.get_json()
    if not data or 'userid' not in data or 'cost_limit' not in data:
        return jsonify({"error": "Missing 'userid' or 'cost_limit' in request body."}), 400

    userid = data['userid']
    try:
        cost_limit = float(data['cost_limit'])
    except ValueError:
        return jsonify({"error": "Invalid cost_limit value. Must be a number."}), 400

    # Fetch user's purchase history
    stock_items = fetch_stock_data(userid)
    if not stock_items:
        return jsonify({"error": "No purchase history found for this user."}), 404

    # Calculate essential items with LLM integration
    essential_list, total_cost, method_used = calculate_essential_items(stock_items, cost_limit, userid)

    return jsonify({
        "essential_items": essential_list,
        "total_cost": total_cost,
        "cost_limit": cost_limit,
        "prioritization_method": method_used
    })

def calculate_essential_items(items, cost_limit, userid):
    """Enhanced calculation with LLM integration and fallback"""
    try:
        # Get user context from MongoDB
        user_data = db.users.find_one({"_id": ObjectId(userid)}) or {}
        health_conditions = user_data.get('current_status', {}).get('health_conditions', [])
        dietary_restrictions = user_data.get('diet', [])
        shopping_habits = user_data.get('shopping_habits', {})
        
        # Prepare LLM prompt
        prompt = f"""Analyze this purchase history and user profile to create a prioritized essential items list within ${cost_limit}.
User Profile:
- Health conditions: {health_conditions}
- Dietary restrictions: {dietary_restrictions}
- Shopping frequency: {shopping_habits.get('frequency', 'normal')}

Recent purchases (last 3 months):
{json.dumps([item for item in items if is_recent(item.get('date', ''))][:15], indent=2)}

Output requirements:
1. JSON array sorted by priority
2. Each entry must contain:
   - item_name (exact product names from purchase history)
   - recommended_quantity (numeric value)
   - priority_reason (short health/nutrition justification)
   - estimated_unit_price (reasonable price estimate)

Prioritization criteria:
- Nutritional needs (especially for {health_conditions})
- Purchase frequency patterns
- Price stability
- Shelf life
- Dietary compatibility with {dietary_restrictions}
"""

        # Get LLM analysis
        response = client.invoke([{"role": "user", "content": prompt}])
        llm_output = response.content.strip().replace("```json", "").replace("```", "").strip()
        
        # Validate and process LLM response
        if validate_llm_output(llm_output):
            llm_items = json.loads(llm_output)
            results, total = process_llm_recommendations(llm_items, items, cost_limit)
            return results, total, "LLM-enhanced"
            
    except Exception as e:
        print(f"LLM integration failed: {str(e)}")

    # Fallback to algorithmic approach
    results, total = algorithmic_fallback(items, cost_limit)
    return results, total, "algorithmic"

def validate_llm_output(data):
    try:
        items = json.loads(data)
        required_keys = {'item_name', 'recommended_quantity', 'priority_reason', 'estimated_unit_price'}
        return all(all(key in item for key in required_keys) for item in items)
    except:
        return False

def process_llm_recommendations(llm_items, historical_items, cost_limit):
    # Create price database from historical data
    price_db = {}
    for item in historical_items:
        if item['quantity'] > 0:
            price_db[item['name']] = item['price'] / item['quantity']

    essential_items = []
    total_cost = 0.0

    for item in llm_items:
        if total_cost >= cost_limit:
            break

        name = item['item_name']
        recommended_qty = float(item['recommended_quantity'])
        estimated_price = float(item['estimated_unit_price'])
        
        # Use historical price if available
        unit_price = price_db.get(name, estimated_price)
        
        if unit_price <= 0:
            continue

        max_affordable = min(
            recommended_qty,
            (cost_limit - total_cost) / unit_price
        )

        if max_affordable > 0:
            essential_items.append({
                "item": name,
                "quantity": round(max_affordable, 2),
                "unit_price": round(unit_price, 2),
                "total_cost": round(max_affordable * unit_price, 2),
                "priority_reason": item['priority_reason']
            })
            total_cost += max_affordable * unit_price

    return essential_items, round(total_cost, 2)

def algorithmic_fallback(items, cost_limit):
    # Original algorithmic implementation
    grouped = defaultdict(lambda: {'total': 0, 'count': 0, 'prices': []})
    
    for item in items:
        name = item['name']
        grouped[name]['total'] += item['quantity']
        grouped[name]['count'] += 1
        if item['quantity'] > 0:
            grouped[name]['prices'].append(item['price'] / item['quantity'])
    
    processed = []
    for name, data in grouped.items():
        avg_qty = data['total'] / data['count'] if data['count'] > 0 else 0
        avg_price = sum(data['prices'])/len(data['prices']) if data['prices'] else 0
        processed.append((name, avg_qty, avg_price))
    
    # Sort by purchase frequency then total quantity
    processed.sort(key=lambda x: (-grouped[x[0]]['count'], -grouped[x[0]]['total']))
    
    essential_items = []
    total_cost = 0.0
    
    for name, avg_qty, avg_price in processed:
        if total_cost >= cost_limit:
            break
            
        if avg_price <= 0:
            continue
            
        max_qty = min(avg_qty, (cost_limit - total_cost) / avg_price)
        
        if max_qty > 0:
            essential_items.append({
                "item": name,
                "quantity": round(max_qty, 2),
                "unit_price": round(avg_price, 2),
                "total_cost": round(max_qty * avg_price, 2)
            })
            total_cost += max_qty * avg_price
    
    return essential_items, round(total_cost, 2)

def is_recent(date_str):
    try:
        purchase_date = datetime.fromisoformat(date_str.replace('T', ' '))
        return (datetime.now() - purchase_date).days < 90
    except:
        return False


#-------------------------------------------- insights calculate ------------------------------------------------------------

@app.route('/monthly_analysis', methods=['POST'])
def monthly_analysis():
    from bson import json_util  # For MongoDB data serialization
    import traceback  # For detailed error logging

    data = request.get_json()
    if not data or 'userid' not in data:
        return jsonify({"error": "Missing 'userid' in request body."}), 400

    userid = data.get('userid')
    current_year = datetime.now().year
    year = data.get('year', current_year)

    try:
        # Convert year to integer and validate
        try:
            year = int(year)
            if year < 2000 or year > current_year + 1:
                raise ValueError
        except ValueError:
            return jsonify({"error": f"Invalid year value. Must be between 2000 and {current_year + 1}"}), 400

        # Fetch and validate purchase data
        raw_stock_items = fetch_stock_data(userid)
        if not raw_stock_items:
            return jsonify({"error": "No purchase history found."}), 404

        # Initialize data structures with explicit type conversion
        monthly_data = defaultdict(lambda: {
            'total_spent': 0.0,
            'items': defaultdict(lambda: {'quantity': 0.0, 'total_cost': 0.0})
        })

        valid_items_count = 0
        for item in raw_stock_items:
            try:
                # Validate required fields
                if not all(key in item for key in ['date', 'name', 'price', 'quantity']):
                    print(f"Skipping invalid item: {item}")
                    continue

                # Parse date with multiple format support
                date_str = item['date'].split('T')[0]  # Handle ISO datetime format
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    date_obj = datetime.strptime(date_str, "%m/%d/%Y")

                if date_obj.year != year:
                    continue

                # Convert numeric fields
                price = float(item['price'])
                quantity = float(item['quantity'])
                if price <= 0 or quantity <= 0:
                    continue

                # Process valid item
                month_key = date_obj.strftime("%B")
                item_name = item['name'].strip().title()
                
                monthly_data[month_key]['total_spent'] += price
                monthly_data[month_key]['items'][item_name]['quantity'] += quantity
                monthly_data[month_key]['items'][item_name]['total_cost'] += price
                valid_items_count += 1

            except Exception as e:
                print(f"Error processing item: {str(e)}\nItem: {item}")
                continue

        # Check if we have valid data after processing
        if valid_items_count == 0:
            return jsonify({
                "error": "No valid purchase records found",
                "debug_info": {
                    "received_items": len(raw_stock_items),
                    "requested_year": year,
                    "common_issues": [
                        "Invalid date formats",
                        "Missing required fields",
                        "Non-positive price/quantity values"
                    ]
                }
            }), 400

        # Convert defaultdict to regular dict for JSON serialization
        processed_data = {
            month: {
                'total_spent': round(data['total_spent'], 2),
                'items': {
                    item: {
                        'quantity': round(details['quantity'], 2),
                        'total_cost': round(details['total_cost'], 2)
                    } for item, details in data['items'].items()
                }
            } for month, data in monthly_data.items()
        }

        # Prepare LLM prompt with strict output formatting
        prompt = f"""Analyze this monthly shopping data from {year} and provide:
1. Month-by-month spending breakdown
2. Identified spending patterns
3. Practical savings advice
4. Budget optimization recommendations

Data Format:
{json.dumps(processed_data, indent=2)}

Respond EXACTLY in this JSON format:
{{
    "analysis": [
        {{
            "month": "MonthName",
            "spending_breakdown": {{
                "total": float,
                "category_distribution": {{
                    "top_category": str,
                    "second_category": str,
                    "other_percentage": float
                }}
            }},
            "key_observation": str,
            "cost_saving_tip": str
        }}
    ],
    "overall_analysis": {{
        "total_year_spending": float,
        "average_monthly_spending": float,
        "estimated_potential_savings": float,
        "recommended_actions": [str, str, str]
    }}
}}"""

        # Get and validate LLM response
        try:
            response = client.invoke([{"role": "user", "content": prompt}])
            llm_response = response.content.strip()
            
            # Clean response and parse
            cleaned_response = llm_response.replace("```json", "").replace("```", "").strip()
            parsed_analysis = json.loads(cleaned_response)
            
            # Validate response structure
            required_keys = {'analysis', 'overall_analysis'}
            if not all(key in parsed_analysis for key in required_keys):
                raise ValueError("Invalid LLM response structure")

            # Add metadata and return
            return jsonify({
                "success": True,
                "analysis": parsed_analysis,
                "raw_data": json.loads(json_util.dumps(processed_data)),
                "metadata": {
                    "processed_year": year,
                    "total_months": len(processed_data),
                    "total_items_processed": valid_items_count
                }
            })

        except Exception as llm_error:
            return jsonify({
                "error": "Analysis failed",
                "llm_error": str(llm_error),
                "llm_response": llm_response,
                "processed_data": processed_data
            }), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


#-------------------------------------------- forecast calculate ------------------------------------------------------------

@app.route('/forecast', methods=['POST'])
def shopping_forecast():
    """Predict future shopping costs and trends"""
    try:
        data = request.get_json()
        if not data or 'userid' not in data:
            return jsonify({"error": "Missing required 'userid' field"}), 400

        userid = data['userid']
        forecast_months = int(data.get('months', 3))  # Default 3 months forecast
        confidence_level = float(data.get('confidence', 0.8))  # Default 80% confidence

        # Validate parameters
        if not (1 <= forecast_months <= 12):
            return jsonify({"error": "Can only forecast 1-12 months"}), 400
        if not (0.5 <= confidence_level <= 0.95):
            return jsonify({"error": "Confidence must be between 0.5-0.95"}), 400

        # Get historical data
        raw_items = fetch_stock_data(userid)
        if not raw_items:
            return jsonify({"error": "No purchase history available"}), 404

        # Process into monthly totals
        monthly_totals = defaultdict(float)
        for item in raw_items:
            try:
                date_str = str(item['date']).split('T')[0].split(' ')[0]
                date_obj = parser.parse(date_str, fuzzy=False)
                month_key = date_obj.strftime("%Y-%m")
                monthly_totals[month_key] += float(item['price'])
            except:
                continue

        # Convert to time series
        sorted_months = sorted(monthly_totals.keys())
        if len(sorted_months) < 3:
            return jsonify({"error": "Minimum 3 months history required"}), 400

        # Create numerical sequence for regression
        X = np.arange(len(sorted_months)).reshape(-1, 1)
        y = np.array([monthly_totals[m] for m in sorted_months])

        # Calculate trend using linear regression
        model = LinearRegression()
        model.fit(X, y)
        trend_slope = model.coef_[0]
        last_value = y[-1]

        # Determine trend type
        trend = "stable"
        if abs(trend_slope) > last_value * 0.05:  # 5% change threshold
            trend = "upward" if trend_slope > 0 else "downward"

        # Generate forecast
        forecast = []
        for i in range(1, forecast_months + 1):
            base_value = last_value + trend_slope * i
            uncertainty = base_value * (1 - confidence_level)
            forecast.append({
                "month": (datetime.strptime(sorted_months[-1], "%Y-%m") + relativedelta(months=i)).strftime("%Y-%m"),
                "estimated_cost": round(base_value, 2),
                "confidence_range": [
                    round(base_value - uncertainty, 2),
                    round(base_value + uncertainty, 2)
                ]
            })

        # Prepare LLM analysis prompt
        prompt = f"""Analyze this shopping trend pattern and forecast:
Historical Data:
{json.dumps(monthly_totals, indent=2)}

Trend Analysis:
- Current trend: {trend}
- Last 3 months: {y[-3:]}
- Forecast months: {forecast_months}

Provide:
1. Plain English trend explanation
2. Key factors influencing the forecast
3. Recommended budgeting strategy

Return JSON format:
{{
    "trend_analysis": {{
        "pattern": "upward/stable/downward",
        "confidence": "high/medium/low",
        "key_drivers": [str, str, str]
    }},
    "recommendations": {{
        "budgeting": [str, str],
        "shopping": [str, str]
    }},
    "risk_factors": [str, str]
}}"""

        # Get LLM insights
        try:
            response = client.invoke([{"role": "user", "content": prompt}])
            llm_analysis = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        except Exception as e:
            llm_analysis = {"error": str(e)}

        return jsonify({
            "forecast": forecast,
            "trend": {
                "type": trend,
                "strength": abs(round(trend_slope / last_value, 2)) if last_value != 0 else 0,
                "confidence": confidence_level
            },
            "analysis": llm_analysis,
            "metadata": {
                "historical_months": len(sorted_months),
                "average_monthly": round(np.mean(y), 2),
                "analysis_model": "Linear Regression + LLM Insights"
            }
        })

    except Exception as e:
        return jsonify({
            "error": "Forecast failed",
            "details": str(e)
        }), 500


#-------------------------------------------- login and register ------------------------------------------------------------


# @app.route('/register', methods=['POST'])
# def register():
#     """
#     Flask endpoint to register a new user with password hashing.
#     """
#     data = request.get_json()
#     required_fields = ['username', 'email', 'password']
#     if not all(field in data for field in required_fields):
#         return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

#     try:
#         # Check if username already exists
#         existing_user = db.users.find_one({"username": data['username']})
#         if existing_user:
#             return jsonify({"error": "Username already exists"}), 400

#         # Hash password
#         hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

#         # Insert new user
#         db.users.insert_one({
#             "username": data['username'],
#             "email": data['email'],
#             "password": hashed_password,
#             "age": data.get('age'),
#             "blood_type": data.get('blood Type'),
#             "diet": data.get('diet'),
#             "habits": data.get('habbits'),
#             "current_status": data.get('current status', {}),
#             "detailed_history": data.get('detailed history', {}),
#             "vaccination_records": data.get('vaccination records', []),
#             "created_at": time.time()
#         })
#         return jsonify({"message": "User registered successfully"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/register', methods=['POST'])
def register():
    """
    Secure user registration with password hashing and data validation
    """
    data = request.get_json()
    required_fields = ['username', 'email', 'password']
    
    # Validate required fields
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

    try:
        # Check for existing user with username or email
        if db.users.find_one({"$or": [
            {"username": data['username']},
            {"email": data['email']}
        ]}):
            return jsonify({"error": "Username or email already exists"}), 409

        # Create user document with hashed password
        user_doc = {
            "username": data['username'],
            "email": data['email'].lower(),
            "password": generate_password_hash(data['password'], method='scrypt'),
            "age": data.get('age'),
            "blood_type": data.get('blood_type'),
            "diet": data.get('diet', []),
            "habits": data.get('habits', []),
            "current_status": data.get('current_status', {}),
            "detailed_history": data.get('detailed_history', {}),
            "vaccination_records": data.get('vaccination_records', []),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Insert into database
        result = db.users.insert_one(user_doc)
        return jsonify({
            "message": "User registered successfully",
            "user_id": str(result.inserted_id)
        }), 201

    except Exception as e:
        return jsonify({"error": "Registration failed", "details": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """
    Secure login implementation with password verification
    """
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Username and password required"}), 400

    try:
        # Find user by username
        user = db.users.find_one({"username": data['username']})
        if not user:
            # Generic error message to prevent username enumeration
            return jsonify({"error": "Invalid credentials"}), 401

        # Verify password
        if check_password_hash(user['password'], data['password']):
            return jsonify({
                "message": "Login successful",
                "user_id": str(user['_id']),
                "username": user['username'],
                "email": user['email']
            }), 200

        return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"error": "Login failed", "details": str(e)}), 500


#-------------------------------------------- monthly purchase analysis ------------------------------------------------------------


@app.route('/monthly_purchase_analysis', methods=['POST'])
def monthly_purchase_analysis():
    """
    Analyze monthly purchase patterns and provide AI insights
    Returns JSON with monthly analysis including trend classification and advice
    """
    data = request.get_json()
    if not data or 'userid' not in data:
        return jsonify({"error": "Missing userid in request"}), 400

    try:
        # Fetch and process purchase data
        stock_items = fetch_stock_data(data['userid'])
        if not stock_items:
            return jsonify({"error": "No purchase history found"}), 404

        # Organize data by month with proper datetime handling
        monthly_data = defaultdict(lambda: {'total': 0.0, 'items': defaultdict(float)})
        
        for item in stock_items:
            try:
                # Handle both string and datetime formats
                date_value = item.get('date')
                if not date_value:
                    continue

                # Parse different date formats
                if isinstance(date_value, datetime):
                    date_obj = date_value
                else:
                    # Handle string dates
                    if 'T' in date_value:
                        date_obj = datetime.fromisoformat(date_value.replace('Z', ''))
                    else:
                        date_obj = datetime.strptime(str(date_value).split()[0], "%Y-%m-%d")
                
                month_str = date_obj.strftime("%B %Y")
                
                # Process numerical values
                price = float(item.get('price', 0))
                quantity = float(item.get('quantity', 0))
                
                monthly_data[month_str]['total'] += price
                monthly_data[month_str]['items'][item.get('name', 'Unknown')] += quantity
                
            except (KeyError, ValueError, AttributeError) as e:
                print(f"Skipping invalid item: {item} - Error: {str(e)}")
                continue

        # Convert to sorted list of months
        sorted_months = sorted(
            monthly_data.keys(),
            key=lambda x: datetime.strptime(x, "%B %Y")
        )

        # Calculate month-over-month changes
        analysis_data = []
        prev_total = None
        for month in sorted_months:
            current_total = monthly_data[month]['total']
            
            # Calculate percentage change
            if prev_total is not None and prev_total != 0:
                pct_change = ((current_total - prev_total) / prev_total) * 100
            else:
                pct_change = 0.0
            
            analysis_data.append({
                "month": month,
                "total_spent": round(current_total, 2),
                "percentage_change": round(pct_change, 1),
                "top_items": dict(sorted(
                    monthly_data[month]['items'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3])
            })
            prev_total = current_total

        # Generate LLM analysis
        prompt = f"""Analyze this monthly purchase data and provide:
1. For each month: 
   - A 3-5 word trend classification (e.g., "Cost Effective Purchases", "Increased Spending")
   - Brief practical advice based on the trend
2. Overall assessment of spending patterns

Data Format:
{json.dumps(analysis_data, indent=2)}

Return JSON format:
{{
    "monthly_analysis": [
        {{
            "month": "Month Year",
            "trend_classification": "string",
            "advice": "string",
            "key_statistics": {{
                "total_spent": float,
                "percentage_change": float,
                "most_purchased_item": "string"
            }}
        }}
    ],
    "overall_assessment": {{
        "spending_trend": "upward/stable/downward",
        "recommendations": ["string", "string"]
    }}
}}"""

        response = client.invoke([{"role": "user", "content": prompt}])
        llm_response = response.content.strip()
        
        # Clean and parse response
        for fmt in ["```json", "```JSON", "```"]:
            llm_response = llm_response.replace(fmt, "")
            
        try:
            parsed_response = json.loads(llm_response)
            return jsonify(parsed_response)
            
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {llm_response}")
            return jsonify({
                "error": "Analysis failed",
                "details": "Could not parse AI response",
                "raw_response": llm_response
            }), 500

    except Exception as e:
        return jsonify({
            "error": "Analysis failed",
            "details": str(e),
            "stack_trace": traceback.format_exc()
        }), 500



#-------------------------------------------- forecast_trends ------------------------------------------------------------

@app.route('/forecast_trends', methods=['POST'])
def forecast_trends():
    """
    Analyze purchase history and forecast future trends
    Returns JSON with:
    - Monthly trend classifications
    - Forecast predictions
    - LLM-generated insights
    """
    data = request.get_json()
    if not data or 'userid' not in data:
        return jsonify({"error": "Missing userid in request"}), 400

    try:
        # Get historical data
        stock_items = fetch_stock_data(data['userid'])
        if not stock_items:
            return jsonify({"error": "No purchase history found"}), 404

        # Process monthly totals with enhanced date handling
        monthly_totals = defaultdict(float)
        monthly_counts = defaultdict(int)
        
        for item in stock_items:
            try:
                # Handle multiple date formats
                date_value = item.get('date')
                if isinstance(date_value, datetime):
                    month_key = date_value.strftime("%Y-%m")
                else:
                    date_str = str(date_value).split('T')[0]
                    month_key = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m")
                
                monthly_totals[month_key] += float(item.get('price', 0))
                monthly_counts[month_key] += 1
            except Exception as e:
                print(f"Skipping invalid item: {e}")
                continue

        # Require at least 3 months of data
        if len(monthly_totals) < 3:
            return jsonify({"error": "Minimum 3 months of data required"}), 400

        # Prepare time series data
        sorted_months = sorted(monthly_totals.keys())
        X = np.arange(len(sorted_months)).reshape(-1, 1)  # Months as ordinal
        y = np.array([monthly_totals[m] for m in sorted_months])  # Monthly totals

        # Calculate trend using linear regression
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        last_value = y[-1]

        # Classify trend
        trend = "stable"
        if abs(slope) > last_value * 0.05:  # 5% threshold
            trend = "upward" if slope > 0 else "downward"

        # Generate forecast for next 3 months
        forecast_months = 3
        forecast = []
        for i in range(1, forecast_months+1):
            forecast.append({
                "month": (datetime.strptime(sorted_months[-1], "%Y-%m") 
                          + relativedelta(months=i)).strftime("%Y-%m"),
                "predicted": round(last_value + slope * i, 2)
            })

        # Prepare LLM prompt
        prompt = f"""Analyze this purchase trend data and provide:
1. Trend classification (upward/stable/downward)
2. 3-month forecast commentary
3. Key factors influencing the trend
4. Practical shopping recommendations

Data:
{json.dumps({
    "historical": [{"month": m, "total": monthly_totals[m]} for m in sorted_months],
    "statistics": {
        "slope": slope,
        "last_month": sorted_months[-1],
        "last_amount": last_value
    }
}, indent=2)}

Return JSON format:
{{
    "trend_analysis": {{
        "classification": "string",
        "confidence": "high/medium/low",
        "key_drivers": ["string", "string"]
    }},
    "forecast": [
        {{
            "month": "YYYY-MM",
            "prediction": "string",
            "recommended_action": "string"
        }}
    ],
    "shopping_advice": {{
        "do": ["string", "string"],
        "avoid": ["string"]
    }}
}}"""

        # Get LLM analysis
        response = client.invoke([{"role": "user", "content": prompt}])
        llm_response = response.content.strip()
        
        # Clean and parse response
        llm_response = llm_response.replace("```json", "").replace("```", "")
        try:
            analysis = json.loads(llm_response)
            
            # Combine statistical and LLM analysis
            return jsonify({
                "statistical_analysis": {
                    "trend": trend,
                    "slope_per_month": round(float(slope), 2),
                    "last_month_value": round(last_value, 2)
                },
                "ai_analysis": analysis,
                "forecast_months": forecast
            })
            
        except json.JSONDecodeError:
            return jsonify({
                "error": "Analysis failed",
                "llm_response": llm_response,
                "statistical_trend": trend
            }), 500

    except Exception as e:
        return jsonify({
            "error": "Forecast failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
