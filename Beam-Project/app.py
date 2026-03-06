from flask import Flask, request, jsonify, render_template
from analysis_engine import analyze_custom_beam
from comparator_engine import ComparatorEngine
import os
from google import genai
from google.genai import types


app = Flask(__name__)

# Initialize Comparator with the CSV file
# Ensure 'Beam Data - Sheet3.csv' is in the main folder
csv_path = 'Beam Data - Sheet3.csv'
if os.path.exists(csv_path):
    comparator = ComparatorEngine(csv_path)
else:
    print(f"WARNING: {csv_path} not found. Comparison features will fail.")
    comparator = None

#configuring gemini API
# The client automatically looks for the GEMINI_API_KEY environment variable.
os.environ["GEMINI_API_KEY"] = "AIzaSyDWPwUPpDRe-iOx5tn2T_d97noxxHTYN1Q"
client = genai.Client()

#Defining the strict System Prompt. NEED TO PLAY WITH THIS
SYSTEM_PROMPT = """
You are an expert Structural Engineering Teaching Assistant. Your goal is to help novice engineering students understand beam design, statics, and material mechanics.
You have access to the user's current beam analysis and comparator results. Use this data to provide specific, highly accurate advice.
Rules:
1. NEVER do the work for them; guide them to the answer.
2. Keep explanations concise, professional, and educational.
3. If asked about a topic unrelated to structural engineering, mechanics, or this web application, politely decline and steer the conversation back to beams.
4. If deflection is high (> L/250), suggest increasing the Moment of Inertia (I) or changing the material (E).
5. If the safety factor is low, suggest a beam with a higher yield stress or a larger section modulus.
"""

@app.route('/')
def home():
    # Looks for index.html inside the 'templates' folder
    return render_template('index.html')

#api call for chat
@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    try:
        data = request.json
        chat_history = data.get('history', [])
        user_message = data.get('message', '')
        context_data = data.get('context', None)

        #Formatting the history for the new Google GenAI SDK
        formatted_history = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append(
                types.Content(
                    role=role, 
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )

        # 2. Inject the structural context into the user's current message
        enhanced_prompt = user_message
        if context_data:
            enhanced_prompt = f"CURRENT APP STATE (Do not mention this raw data to the user, just use it to answer their question):\n{context_data}\n\nUSER'S ACTUAL QUESTION:\n{user_message}"

        # 3. Create the chat session with the system instructions and history
        chat = client.chats.create(
            model="gemini-2.5-flash", # Using the latest, fastest model
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7, # Keeps responses focused but conversational
            ),
            history=formatted_history
        )
        
        # 4. Send the message to the LLM
        response = chat.send_message(enhanced_prompt)

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"AI Chat Error: {e}")
        return jsonify({"error": "Failed to connect to the AI Assistant. Please check server logs."}), 500

#beam analysis API
@app.route('/api/analyse', methods=['POST'])
def analyse_manual():
    try:
        data = request.json
        # The frontend sends 'beam_weight', analysis engine expects it.
        # Everything else (E, I, loads, supports) should match keys.
        results = analyze_custom_beam(data)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#beam recommendation api
@app.route('/api/recommend', methods=['POST'])
def recommend_beam():
    if not comparator:
        return jsonify({"error": "Database not loaded"}), 500
    try:
        data = request.json
        # Clean up data if necessary (e.g., ensure strings are lower case)
        results = comparator.recommend(data)
        return jsonify(results)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

#loading database into website   
@app.route('/api/database', methods=['GET'])
def get_database():
    if not comparator:
        return jsonify({"error": "Database not loaded on server."}), 500
    
    try:
        #Accessing the dataframe already loaded in memory
        df = comparator.beam_library
        
        #Extracting headers
        headers = df.columns.tolist()
        
        #Replace NaN or missing values with empty strings so JSON doesn't break
        df_clean = df.fillna("") 
        
        #Extract rows as a 2D list
        rows = df_clean.values.tolist()
        
        return jsonify({
            "headers": headers,
            "rows": rows
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)