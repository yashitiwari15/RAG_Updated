# from flask import Blueprint, request, jsonify
# from pymilvus import connections, Collection, utility, MilvusException

# delete_bp = Blueprint('delete_bp', __name__)

# def connect_to_milvus():
#     try:
#         connections.connect("default", host="0.0.0.0", port="19530")
#         if utility.has_collection("exmpcollection1"):
#             return Collection("exmpcollection1")
#         return None
#     except MilvusException as e:
#         print(f"Failed to connect to Milvus: {e}")
#         return None

# collection = connect_to_milvus()

# @delete_bp.route("/delete", methods=["POST"])
# def delete_pdf():
#     try:
#         data = request.get_json()
#         filename = data.get("filename", "").lower()  # Get the filename from the request data and convert to lowercase

#         if collection is None:
#             return jsonify({"error": "Failed to connect to Milvus."}), 500

#         if not filename:
#             return jsonify({"error": "Filename must be provided."}), 400

#         # Delete all records related to the given filename
#         filter_expr = f'pdf_name == "{filename}"'
#         result = collection.delete(expr=filter_expr)

#         return jsonify({"message": f"Records related to filename '{filename}' have been deleted."}), 200

#     except MilvusException as e:
#         return jsonify({"error": f"Milvus Error: {e}"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
from flask import Blueprint, request, jsonify
from pymilvus import connections, Collection, utility, MilvusException

delete_bp = Blueprint('delete_bp', __name__)

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

@delete_bp.route("/delete", methods=["POST"])
def delete_pdf():
    try:
        data = request.get_json()
        filename = data.get("filename", "").lower()  # Get the filename from the request data and convert to lowercase

        if collection is None:
            return jsonify({"error": "Failed to connect to Milvus."}), 500

        if not filename:
            return jsonify({"error": "Filename must be provided."}), 400

        # Check if any records exist with the given filename
        filter_expr = f'pdf_name == "{filename}"'
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        search_results = collection.query(expr=filter_expr)

        if not search_results:
            return jsonify({"message": f"No records found for filename '{filename}'. The file may have already been deleted."}), 404

        # Delete all records related to the given filename
        result = collection.delete(expr=filter_expr)

        return jsonify({"message": f"Records related to filename '{filename}' have been deleted."}), 200

    except MilvusException as e:
        return jsonify({"error": f"Milvus Error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
