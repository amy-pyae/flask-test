from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from bson import ObjectId
import hashlib


app = Flask(__name__)
load_dotenv()
api_key = os.getenv("API_KEY")
# Configure the Flask app to use your MongoDB URI from the .env file.
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
# Initialize PyMongo with the Flask app
mongo = PyMongo(app)

def generate_key(data: str):
    return hashlib.sha256(data.encode()).hexdigest()[:16]  # Fixed 16 chars

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask MongoDB App!"})

# Create a new agent writer
@app.route('/api/v1/agent/persona/create', methods=['POST'])
def create_agent_writer():
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'industry', 'instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Insert the new document into the agent_writers collection
    result = mongo.db.agent_writers.insert_one({
        "name": data['name'],
        "industry": data['industry'],
        "instruction": data['instruction'],
        "prompt": data['instruction']
    })
    return jsonify({"id": str(result.inserted_id)}), 201

# Create a new agent task
@app.route('/api/v1/agent/task/create', methods=['POST'])
def create_agent_task():
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'goal', 'instruction', 'is_writer_need', 'default_writer']

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    original_instruction = data['instruction']
    persona= "${persona}"
    prompt = data['instruction']
    role = "admin"

    if data['is_writer_need']:
        prompt = persona + "\n\n" + original_instruction  # Make sure this is indented

    if data['goal']:
        prompt = data['goal'] + "\n\n" + prompt  # Make sure this is indented

    default_writer_input = data.get('default_writer')
    if default_writer_input is None or default_writer_input.strip() == "":
        writer_obj_id = None
    else:
        # Attempt to convert the provided writer id to an ObjectId.
        try:
            writer_obj_id = ObjectId(default_writer_input)
        except Exception as e:
            return jsonify({"error": "Invalid default_writer id"}), 400

        # Optionally check that the writer exists in the agent_writers collection.
        writer = mongo.db.agent_writers.find_one({"_id": writer_obj_id})
        if writer is None:
            return jsonify({"error": "Default writer not found"}), 404

    # Insert the new document into the agent_writers collection
    result = mongo.db.agent_tasks.insert_one({
        "name": data['name'],
        "key":generate_key(data['name']),
        "goal": data['goal'],
        "instruction": original_instruction,
        "prompt": prompt,
        "is_writer_need": data['is_writer_need'],
        "default_writer": writer_obj_id,
        "parameters": data['parameters'],
        "role":"admin",     # admin,developer
                            # "created_by":"user"
    })
    return jsonify({"id": str(result.inserted_id)}), 201

@app.route('/api/v1/agent/task/all', methods=['GET'])
def get_agent_tasks():
    tasks = list(mongo.db.agent_tasks.aggregate([
        {"$lookup": {
            "from": "agent_writers",
            "localField": "default_writer",
            "foreignField": "_id",
            "as": "agent_writer"
        }},
        {"$unwind": {
            "path": "$agent_writer",
            "preserveNullAndEmptyArrays": True
        }},
        {"$project": {
            "name": 1,
            "goal": 1,
            "instruction": 1,
            "is_writer_need": 1,
            "parameters":1,
            "agent_writer": "$agent_writer.name"
        }}
    ]))
    for t in tasks:
        t["_id"] = str(t["_id"])
    return jsonify(tasks)

@app.route('/api/v1/agent/task', methods=['GET'])
def get_agent_tasks_by_name():
    key = request.args.get("key")
    pipeline = [
        {"$match": {"key": key}},
        {"$lookup": {
            "from": "agent_writers",
            "localField": "default_writer",
            "foreignField": "_id",
            "as": "agent_writer"
        }},
        {"$unwind": {"path": "$agent_writer", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "name": 1,
            "goal": 1,
            "instruction": 1,
            "is_writer_need": 1,
            "parameters": 1,
            "agent_writer": "$agent_writer.name"
        }}
    ]

    result = list(mongo.db.agent_tasks.aggregate(pipeline))
    if not result:
        return jsonify({"error": "Agent task not found"}), 404

    task = result[0]
    task["_id"] = str(task["_id"])
    return jsonify(task)


if __name__ == '__main__':
    app.run(debug=True)
