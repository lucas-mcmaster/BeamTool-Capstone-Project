from flask import Flask, request, jsonify, render_template
from analysis_engine import analyze_custom_beam
from comparator_engine import ComparatorEngine
import os

app = Flask(__name__)

# Initialize Comparator with the CSV file
# Ensure 'Beam Data - Sheet3.csv' is in the main folder
csv_path = 'Beam Data - Sheet3.csv'
if os.path.exists(csv_path):
    comparator = ComparatorEngine(csv_path)
else:
    print(f"WARNING: {csv_path} not found. Comparison features will fail.")
    comparator = None

@app.route('/')
def home():
    # Looks for index.html inside the 'templates' folder
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)