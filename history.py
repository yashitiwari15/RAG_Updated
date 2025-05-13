import os
import json
from flask import Blueprint,jsonify, request

history_bp = Blueprint('history_bp', __name__)

@history_bp.route("/history", methods=["GET"])
def get_user_history():
    try:
        # Get the user_id from the request query parameters
        user_id = request.args.get("user_id", None)

        if not user_id:
            return jsonify({"error": "user_id parameter is missing"}), 400

        # Path to the interactions.json file
        output_dir = "interactions"
        filepath = os.path.join(output_dir, "interactions.json")

        # Check if the interactions.json file exists
        if not os.path.exists(filepath):
            return jsonify({"error": "No interactions found"}), 404

        # Read the interactions.json file
        with open(filepath, 'r') as json_file:
            try:
                interactions = json.load(json_file)
            except json.JSONDecodeError:
                return jsonify({"error": "Failed to read interactions file"}), 500

        # Find the interaction history for the given user_id
        for interaction in interactions:
            if interaction["user_id"] == user_id:
                return jsonify({
                    "user_id": user_id,
                    "chat_history": interaction["chat_history"]
                })

        # If no interaction found for the user_id, return an error
        return jsonify({"error": f"No chat history found for user_id: {user_id}"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
