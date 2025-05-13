from flask import Flask
from uploadnew import upload_bp
from query5 import query_bp
from filenames import filenames_bp
from delete import delete_bp
from clear import clear_bp  
from history import history_bp
#from highlight import highlight_bp  # Import the new highlight blueprint
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv(".env")

# Register blueprints
app.register_blueprint(upload_bp)
app.register_blueprint(query_bp)
app.register_blueprint(filenames_bp)
app.register_blueprint(delete_bp)
app.register_blueprint(clear_bp)
app.register_blueprint(history_bp)
#app.register_blueprint(highlight_bp)  # Register the highlight blueprint

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
