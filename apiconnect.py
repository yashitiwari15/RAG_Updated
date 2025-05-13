from flask import Flask
from upload import upload_bp
from query import query_bp
from filenames import filenames_bp
from delete import delete_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(upload_bp)
app.register_blueprint(query_bp)
app.register_blueprint(filenames_bp)
app.register_blueprint(delete_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
