from flask import Blueprint, request, jsonify
import fitz  # PyMuPDF
import os

highlight_bp = Blueprint('highlight_bp', __name__)

import fitz  # PyMuPDF
import os

def highlight_text_in_pdf(filename, highlights):
    """
    Highlights text in a PDF based on the provided highlights.

    :param filename: The filename of the PDF to be highlighted
    :param highlights: A list of dictionaries containing 'text', 'page_number', and 'filename'
    :return: The path to the saved, highlighted PDF file
    """
    pdf_path = os.path.join("path_to_your_pdfs", filename)
    
    doc = fitz.open(pdf_path)

    for highlight in highlights:
        page_number = highlight['page_number'] - 1  # Pages are zero-indexed in PyMuPDF
        text_to_highlight = highlight['text']

        page = doc[page_number]
        text_instances = page.search_for(text_to_highlight)

        for inst in text_instances:
            page.add_highlight_annot(inst)

    output_path = f"highlighted_{os.path.basename(pdf_path)}"
    doc.save(output_path, garbage=4, deflate=True)
    return output_path


@highlight_bp.route("/highlight", methods=["POST"])
def highlight_pdf():
    try:
        data = request.get_json()
        highlights = data.get("highlights", [])
        filename = data.get("filename", "")

        if not highlights or not filename:
            return jsonify({"error": "Missing highlights or filename in the request"}), 400

        pdf_path = os.path.join("path_to_your_pdfs", filename)  # Update with your PDF storage path
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"File {filename} not found"}), 404

        # Highlight the PDF
        output_pdf_path = highlight_text_in_pdf(pdf_path, highlights)

        # Return the path to the highlighted PDF
        return jsonify({"message": "PDF highlighted successfully", "pdf_path": output_pdf_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
