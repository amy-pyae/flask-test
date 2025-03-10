from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("API_KEY")


# Configure the Flask app to use your MongoDB URI from the .env file.
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

# Initialize PyMongo with the Flask app
mongo = PyMongo(app)

from db import db

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask MongoDB App!"})


@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    try:
        # Try listing collections in the database as a simple connection test
        collections = mongo.db.list_collection_names()
        return jsonify({
            "status": "success",
            "collections": collections
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    # Query the 'projects' collection from the connected database
    projects = list(mongo.db.projects.find())

    # Convert ObjectId fields to string for JSON serialization
    for project in projects:
        project["_id"] = str(project["_id"])

    return jsonify(projects), 200


if __name__ == '__main__':
    app.run(debug=True)
