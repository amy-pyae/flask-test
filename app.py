from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from config import Config
from extensions import mongo


load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

mongo.init_app(app)


from routes.writers import seo_writer_bp
from routes.projects import project_bp
from routes.agents import agent_bp
from routes.chatbot import chatbot_bp
from routes.generate_blog import generate_blog_bp
from routes.unstructure_upload import un_structure_bp

API_PREFIX = '/api'
app.register_blueprint(seo_writer_bp, url_prefix=API_PREFIX)
app.register_blueprint(project_bp, url_prefix=API_PREFIX)
app.register_blueprint(agent_bp, url_prefix=API_PREFIX)
app.register_blueprint(chatbot_bp, url_prefix=API_PREFIX)
app.register_blueprint(generate_blog_bp, url_prefix=API_PREFIX)
app.register_blueprint(un_structure_bp, url_prefix=API_PREFIX)


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask MongoDB App!"})


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)