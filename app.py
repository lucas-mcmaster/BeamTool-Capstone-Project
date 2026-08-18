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

#configuring gemini API for web testing 
# 1. Store all your backup keys in a list
API_KEYS = [
    "AIzaSyBvE4j0TGjVKSxDPCasxUYZ_KIvjkVC0aA", 
    "AIzaSyDWPwUPpDRe-iOx5tn2T_d97noxxHTYN1Q",
    "AIzaSyD3dHX64m9xwoV3oqSbFHRy6784cFy7bbU",
    "AIzaSyBYHNH6oMYyXHm3SVhnykusvJK0B7w7Bac"
]

# 2. Track the current key index
current_key_idx = 0

# 3. Initialize the client with the first key
client = genai.Client(api_key=API_KEYS[current_key_idx])

#Defining the strict System Prompt. NEED TO PLAY WITH THIS
SYSTEM_PROMPT = """
You are an expert Structural Engineering Teaching Assistant. Your goal is to guide novice engineering students in understanding statics, beam design, and material mechanics.

You will receive a hidden "CURRENT APP STATE" containing the user's live inputs and their latest analysis or comparator results. Use this data to provide contextual, highly specific advice.

PLATFORM ARCHITECTURE & FUNCTIONALITY:
1. Manual Analysis Tab: This is a direct beam solver which uses Macaulay's method. Users input specific 'E' (Stiffness) and 'I' (Geometry) values to see exact Shear, Moment, and Deflection diagrams for a single beam.
2. Design Mode Tab: This is a design recommendation tool. It runs a Multi-Criteria Decision Analysis. It filters the internal database for safety/environment constraints, then scores candidates using a weighted average of Cost, Weight, and Safety. Cost and weight have a higher importance, but the user is also able to include optional optimization preferences which adjust the importance of each factor. Candidate beams with a safety factor below 1 or deflection greater than L/240 are automatically rejected, but this is not shown to the user so if they ask you should explain this to them. 
3. Design History: Designs from Design Mode Tab are automatically saved in this tab (to the user's browser 'localStorage'). They are private to the device and will be lost if the browser cache is cleared. Users can 'Restore' previous designs to the Comparator inputs.
4. Beam Database: A searchable library of real-world sections for different materials (W-shapes, HSS, etc.). The AI can encourage users to check the database to see available material properties.
5. Visualizer: The SVG at the top of the results is a live structural model. Red arrows are loads, and the supports are shown with standard images for a pin, roller and fixed support.

CORE TEACHING GOALS:
1. Prevent Overdesign: Novice engineers often select massively oversized beams "just to be safe." If you see the user select a minimum Safety Factor (FoS) greater than 3.0, gently challenge the user to consider a lighter or cheaper alternative to optimize their design. Teach them that good engineering is about efficiency, not just raw strength.
2. Explain Trade-offs: In Recommendation/Comparator Mode, use the "OTHER CANDIDATES" data to explain *why* the engine picked the winner. Compare cost vs. safety factor vs. deflection to teach engineering judgment (e.g., "Candidate 2 is $50 cheaper, but notice how its deflection is dangerously close to the limit (L/240)").
3. Connect Physics to Variables and Results: 
   - If deflection is high, teach them that deflection is controlled by stiffness (Material Modulus 'E' and Geometry 'I'). 
   - If the safety factor is failing, teach them it is controlled by strength (Yield Stress) and Section Modulus.
4. Be a Guiding Expert: As a stuctural engineering expert, if the user asks complex questions give a suitable response. If they ask for your recommendation or help with a question guide them through it with your answer.

STRICT OPERATING RULES:
1. STAY ON TOPIC: If asked about non-engineering topics (e.g., writing essays, general coding, history), politely decline and steer the conversation back to structural mechanics.
2. CONCISENESS: Keep responses short, punchy, and highly readable. You are a chat bot, not a textbook. But avoid using bullet points as they do not show well in the text box.
3. REFERENCING UI: Encourage the user to look at the visual aids on their screen. Reference the Shear Force Diagram, Bending Moment Diagram, and Deflection curves to help them visualize internal forces.
"""

@app.route('/')
def home():
    # Looks for index.html inside the 'templates' folder
    return render_template('index.html')

#api call for chat
@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    global client, current_key_idx
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

        #for loop for API key rotation
        max_attemps=len(API_KEYS)

        # 3. Create the chat session with the system instructions and history
        for attempt in range(max_attemps):
            try:
                chat = client.chats.create(
                    model="gemini-2.5-flash", # Using the latest, fastest model
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.9, # Keeps responses focused but conversational
                    ),
                    history=formatted_history
                )
        
                # 4. Sending the message to the LLM
                response = chat.send_message(enhanced_prompt)
                #printing chat history for cost testing in Google AI Studio
                print("\n--- CHAT HISTORY (Recreate these turns) ---")
                if not chat_history:
                    print("[No history yet - this is the first message]")
                for msg in chat_history:
                    role = "USER" if msg["role"] == "user" else "MODEL (AI)"
                    print(f"{role}: {msg['content']}")
                print("\n--- ENHANCED PROMPT (Context + New Message) ---")
                print(enhanced_prompt)
                print("="*10 + "\n")

                return jsonify({"reply": response.text})

            except Exception as ai_error:
                error_str = str(ai_error).lower()
                # Check if the error is related to rate limits or quota (HTTP 429)
                if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                    print(f"⚠️ API Key {current_key_idx + 1} exhausted! Switching to next key...")
                    
                    # Move to the next key (and loop back to 0 if at the end of the list)
                    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    
                    # Re-initialize the client with the new key
                    client = genai.Client(api_key=API_KEYS[current_key_idx])
                    
                    # The loop will naturally restart and try again with the new client
                    continue 
                else:
                    # If it's a different error (like a network drop), don't burn through keys, just fail
                    raise ai_error
                
        #If the loop finishes without returning, ALL keys are dead      
        return jsonify({"error": "System overloaded: All AI API keys have exhausted their daily quota. Please try again tomorrow."}), 429
     
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
    app.run(host="0.0.0.0", debug=True, port=5001)