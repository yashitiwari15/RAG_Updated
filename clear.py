import os
import json
from flask import Blueprint, jsonify

# Define the Blueprint for the clear route
clear_bp = Blueprint('clear_bp', __name__)

# Assuming chat_history is a global or accessible variable
chat_history = {}  # Reset chat history in memory

@clear_bp.route("/clear", methods=["POST"])
def clear_interactions():
    try:
        # Path to the interactions.json file
        output_dir = "interactions"
        filepath = os.path.join(output_dir, "interactions.json")

        # Check if the file exists and clear it
        if os.path.exists(filepath):
            # Overwrite the file with an empty list
            with open(filepath, 'w') as json_file:
                json.dump([], json_file, indent=2)
        
        # Clear the in-memory chat history as well
        global chat_history
        chat_history.clear()  # Clear all chat history stored in memory

        return jsonify({"message": "Interactions and chat history cleared successfully."})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
