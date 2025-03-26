from helpers.Helper import generate_agent_key
from flask import Blueprint, request, jsonify
from extensions import mongo
from bson import ObjectId

agent_bp = Blueprint('agents', __name__)

@agent_bp.route('/v2/agent/create', methods=['POST'])
def create_agent_v2():
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'goal', 'instruction', 'is_writer_need', 'personas', 'default_persona']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    original_instruction = data['instruction']
    persona = "<<<Persona>>>"
    prompt = data['instruction']
    role = "admin"
    llm = {
        "llm_name": "gemini",
        "model": "gemini-flash-pro"
    }

    if data['is_writer_need']:
        prompt = persona + "\n\n" + original_instruction

    # Process default_writer as a list of writer ids
    default_writer_input = data.get('default_writer')
    writer_obj_ids = []
    if default_writer_input:
        if not isinstance(default_writer_input, list):
            return jsonify({"error": "default_writer should be a list of writer ids"}), 400
        for writer_id in default_writer_input:
            try:
                oid = ObjectId(writer_id)
            except Exception as e:
                return jsonify({"error": f"Invalid default_writer id: {writer_id}"}), 400
            # Optionally, check that the writer exists in the agent_writers collection.
            writer = mongo.db.agent_writers.find_one({"_id": oid})
            if writer is None:
                return jsonify({"error": f"Default writer not found: {writer_id}"}), 404
            writer_obj_ids.append(oid)

    # Insert the new document into the agent_tasks collection
    result = mongo.db.agent_tasks.insert_one({
        "name": data['name'],
        "key": generate_agent_key(data['name']),
        "goal": data['goal'],
        "instruction": original_instruction,
        "prompt": prompt,
        "is_writer_need": data['is_writer_need'],
        "default_writer": writer_obj_ids,  # Now storing a list of ObjectIds
        "personas": data['personas'],
        "parameters": data['parameters'],
        "default_llm": llm,
        "role": "admin"  # admin, developer
    })
    return jsonify({"id": str(result.inserted_id)}), 201

@agent_bp.route('/v1/agent/<id>', methods=['GET'])
def get_agent_task(id):
    try:
        task = mongo.db.agent_tasks.find_one({"_id": ObjectId(id)})
        if task:
            task["_id"] = str(task["_id"])
            return jsonify(task), 200
        else:
            return jsonify({"error": "Agent not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@agent_bp.route('/v1/agent/<id>', methods=['PUT'])
def update_agent_task(id):
    data = request.get_json()
    # Validate required fields (adjust these as needed)
    required_fields = ['name', 'instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Update the agent task document using data from the request.
    result = mongo.db.agent_tasks.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "name": data['name'],
            "instruction": data['instruction'],
            "parameters": data.get('parameters', {}),
            "goal": data.get('goal', ""),
            "is_writer_need": data.get('is_writer_need', False),
            "default_writer": data.get('default_writer', None)
        }}
    )
    if result.matched_count == 0:
        return jsonify({"error": "Agent task not found"}), 404

    return jsonify({"id": id}), 200


@agent_bp.route('/v1/agent/all', methods=['GET'])
def get_agent_tasks():
    tasks = list(mongo.db.agent_tasks.aggregate([
        {"$match": {"role": "admin"}},  # Filter documents where role is "admin"

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
            "key": 1,
            "instruction": 1,
            "is_writer_need": 1,
            "parameters": 1,
            "agent_writer": "$agent_writer.name"
        }}
    ]))
    for t in tasks:
        t["_id"] = str(t["_id"])
    return jsonify(tasks)


@agent_bp.route('/v1/agent', methods=['GET'])
def get_agent_tasks_by_key():
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