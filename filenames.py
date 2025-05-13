from flask import Blueprint, jsonify
from pymilvus import connections, Collection, utility, MilvusException

# Assuming you have already connected to Milvus and created the collection
filenames_bp = Blueprint('filenames_bp', __name__)

def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0", port="19530")
        if utility.has_collection("exmpcollection1"):
            return Collection("exmpcollection1")
        return None
    except MilvusException as e:
        print(f"Failed to connect to Milvus: {e}")
        return None

collection = connect_to_milvus()

@filenames_bp.route("/filenames", methods=["GET"])
def get_filenames():
    try:
        if collection is None:
            return jsonify({"error": "Failed to connect to Milvus."}), 500

        # Query to get all filenames (distinct values of `pdf_name`)
        results = collection.query(
            expr="",
            output_fields=["pdf_name"],
            limit=1000
        )
        
        # Extracting filenames from the results
        filenames = list(set([res["pdf_name"] for res in results]))

        return jsonify({"filenames": filenames}), 200
    except MilvusException as e:
        return jsonify({"error": f"Milvus Error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
