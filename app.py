from flask import Flask, request, jsonify,make_response
from flask_pymongo import PyMongo
from flask_cors import CORS
from dotenv import load_dotenv
from bson import ObjectId
from langchain.prompts import PromptTemplate  # Using LangChain for prompt formatting
import google.generativeai as genai
import hashlib
import os
from string import Formatter
import jwt
import re
import datetime
import json

load_dotenv()

app = Flask(__name__)

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

# Configure API Key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CORS(app)

# Initialize Gemini Pro Chat Model
chat_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_key(data: str):
    return hashlib.sha256(data.encode()).hexdigest()[:16]  # Fixed 16 chars

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask MongoDB App!"})


@app.route('/api/v1/persona/create', methods=['POST'])
def create_agent_writer():
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'industry','goal', 'instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Insert the new document into the agent_writers collection
    result = mongo.db.agent_writers.insert_one({
        "name": data['name'],
        "industry": data['industry'],
        "instruction": data['instruction'],
        "prompt": data['instruction'],
        "created_by": None
    })
    return jsonify({"id": str(result.inserted_id)}), 201

@app.route('/api/v1/persona/<id>', methods=['PUT'])
def update_agent_writer(id):
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'industry', 'goal', 'instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Update the document in the agent_writers collection
    result = mongo.db.agent_writers.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "name": data['name'],
            "industry": data['industry'],
            "goal": data['goal'],
            "instruction": data['instruction'],
            "prompt": data['instruction']  # using instruction as prompt, like in creation
        }}
    )

    if result.matched_count == 0:
        return jsonify({"error": "Persona not found"}), 404

    return jsonify({"id": id}), 200

@app.route('/api/v1/persona/<id>', methods=['GET'])
def get_persona(id):
    try:
        persona = mongo.db.agent_writers.find_one({"_id": ObjectId(id)})
        if persona:
            persona["_id"] = str(persona["_id"])
            return jsonify(persona), 200
        else:
            return jsonify({"error": "Persona not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/persona/all', methods=['GET'])
def get_personal_all():
    personas = list(mongo.db.agent_writers.aggregate([
        # {"$match": {"role": "admin"}},  # Filter documents where role is "persona"
        {"$project": {
            "name": 1,
            "industry": 1,
            "instruction": 1
        }}
    ]))
    for t in personas:
        t["_id"] = str(t["_id"])
    return jsonify(personas)

# Create a new agent task
@app.route('/api/v1/agent/create', methods=['POST'])
def create_agent_task():
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'goal', 'instruction', 'is_writer_need', 'default_writer']

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    original_instruction = data['instruction']
    persona= "<<<Persona>>>"
    prompt = data['instruction']
    role = "admin"
    llm = {
        "llm_name": "gemini",
        "model" : "gemini-flash-pro"
    }

    if data['is_writer_need']:
        prompt = persona + "\n\n" + original_instruction  # Make sure this is indented

    # if data['goal']:
    #     prompt = data['goal'] + "\n\n" + prompt  # Make sure this is indented

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
        "personas": data['personas'],
        "is_writer_need": data['is_writer_need'],
        "default_writer": writer_obj_id,
        "parameters": data['parameters'],
        "default_llm": llm,
        "role":"admin",     # admin,developer
                            # "created_by":"user"
    })
    return jsonify({"id": str(result.inserted_id)}), 201

@app.route('/api/v1/agent/<id>', methods=['GET'])
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

@app.route('/api/v1/agent/<id>', methods=['PUT'])
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

@app.route('/api/v1/agent/all', methods=['GET'])
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
            "key":1,
            "instruction": 1,
            "is_writer_need": 1,
            "parameters":1,
            "agent_writer": "$agent_writer.name"
        }}
    ]))
    for t in tasks:
        t["_id"] = str(t["_id"])
    return jsonify(tasks)

@app.route('/api/v1/agent', methods=['GET'])
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

def text_to_html(text):
    """Converts the given text to HTML format."""

    html = "<html><head><title>Dominate Singapore Search: A Comprehensive Guide to Local Keyword Research</title><meta name=\"description\" content=\"Keyword research is the magic key to unlocking online visibility in Singapore. This comprehensive guide unveils the secrets to potent keyword research, transforming your business's online presence. Discover how to unearth hidden gems, understand search intent, and conquer your niche. Learn strategies for effective keyword implementation and watch your website soar to the top of search results.\"></head><body>"

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            level = line.count("#")
            html += f"<h{level}>{line.lstrip('# ').strip()}</h{level}>"
        elif line.startswith("Q:"):
            html += f"<p><strong>{line}</strong></p>"
        elif line.startswith("A:"):
            html += f"<p>{line}</p>"
        elif line:
            html += f"<p>{line}</p>"

    html += "</body></html>"
    return html

def replace_prompt_variables(prompt, parameters):
    """Replaces placeholders in the prompt using LangChain's PromptTemplate, ensuring values are enclosed in double quotes."""

    # Replace "${" with "{" for compatibility with PromptTemplate
    prompt = prompt.replace("${", "{")

    # Wrap provided parameter values in double quotes
    quoted_parameters = {key: f'"{value}"' for key, value in parameters.items()}

    # Parse the prompt to find all placeholders
    formatter = Formatter()
    placeholders = [field_name for _, field_name, _, _ in formatter.parse(prompt) if field_name]

    # Ensure every placeholder has a value; if missing, default to an empty string.
    for field in placeholders:
        if field not in quoted_parameters:
            quoted_parameters[field] = ""

    # Use LangChain's PromptTemplate for replacement
    prompt_template = PromptTemplate.from_template(prompt)
    return prompt_template.format(**quoted_parameters)

def get_task(task_key):
    task = mongo.db.agent_tasks.find_one(
        {"key": task_key},
        {"instruction":1, "prompt": 1, "is_writer_need": 1, "default_writer": 1, "parameters": 1}
    )

    if task:
        # Convert ObjectId fields to plain strings.
        task["_id"] = str(task["_id"])
        if "default_writer" in task and task["default_writer"]:
            task["default_writer"] = str(task["default_writer"])

    return task

@app.route('/api/v1/prompt', methods=['POST'])
def test_content():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    json_format = data.get('json_format')

    if api_key != os.getenv("SYSTEM_API_KEY"): # need to change
        return jsonify({"error": "Invalid api key"}), 400


    task_key = data.get('key', '').strip()
    print(f"""Here is key {task_key}""")
    parameters = data.get('parameters', {})
    llm = data.get('llm', {})
    # llm = {
    #     "llm_name": "Gemini",
    #     "model": "gemini-1.5-flash"
    # }

    task = get_task(task_key)

    if not task:
        return jsonify({"error": "Task not found"}), 404

    prompt = task["prompt"]
    placeholder = "<<<Persona>>>"
    final_prompt = task["instruction"]

    if task["is_writer_need"]:
        obj_id = None
        if task["default_writer"]:
            try:
                obj_id = ObjectId(task["default_writer"])
            except Exception as e:
                return jsonify({"error": "Invalid selected writer id"}), 400
        else:
            try:
                obj_id = ObjectId(parameters["persona"])
                if "persona" in parameters:
                    del parameters["persona"]
            except Exception as e:
                return jsonify({"error": "Invalid writer id"}), 400

        if obj_id:
            writer = mongo.db.agent_writers.find_one({"_id": obj_id})
            if not writer:
                return jsonify({"error": "Writer not found"}), 404
            final_prompt = prompt.replace(placeholder, writer["prompt"])

    # return final_prompt
    try:
        response = chat_model.generate_content(final_prompt)
    except Exception as e:
            return jsonify({"error": f"Error calling Gemini API: {str(e)}"}), 500

    # html_output = text_to_html(response.text)
    #
    # response = make_response(html_output)
    # response.headers['Content-Type'] = 'text/html'
    if json_format:
        cleaned_json_str = re.sub(r'```json\n|\n```', '', response.text).strip()
        json_data = json.loads(cleaned_json_str)  # Convert string to JSON (Python object)
        return jsonify({"data": json_data}), 200
    else:
        return jsonify({"data": response.text}), 200

from flask import request, jsonify
from bson import ObjectId

from flask import request, jsonify
from bson import ObjectId

@app.route('/api/v2/agent/create', methods=['POST'])
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
        "key": generate_key(data['name']),
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

@app.route('/api/v2/prompt', methods=['POST'])
def prompt_v2():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    json_format = data.get('json_format')
    # return_html = data.get('return_html')

    if api_key != os.getenv("SYSTEM_API_KEY"): # need to change
        return jsonify({"error": "Invalid api key"}), 400


    task_key = data.get('key', '').strip()
    parameters = data.get('parameters', {})
    llm = data.get('llm', {})
    # llm = {
    #     "llm_name": "Gemini",
    #     "model": "gemini-1.5-flash"
    # }

    task = get_task(task_key)

    if not task:
        return jsonify({"error": "Task not found"}), 404

    prompt = task["prompt"]
    placeholder = "<<<Persona>>>"
    final_prompt = task["instruction"]

    if task["is_writer_need"]:
        obj_id = None
        if task["default_writer"]:
            try:
                obj_id = ObjectId(task["default_writer"])
            except Exception as e:
                return jsonify({"error": "Invalid selected writer id"}), 400
        else:
            try:
                obj_id = ObjectId(parameters["persona"])
                if "persona" in parameters:
                    del parameters["persona"]
            except Exception as e:
                return jsonify({"error": "Invalid writer id"}), 400

        if obj_id:
            writer = mongo.db.agent_writers.find_one({"_id": obj_id})
            if not writer:
                return jsonify({"error": "Writer not found"}), 404
            final_prompt = prompt.replace(placeholder, writer["prompt"])

    if len(parameters) > 0:
        if task_key != '474acb4b2ff762b7':
            final_prompt = replace_prompt_variables(final_prompt, parameters)

    try:
        response = chat_model.generate_content(final_prompt)
    except Exception as e:
            return jsonify({"error": f"Error calling Gemini API: {str(e)}"}), 500

    # if return_html:
    #     html_output = text_to_html(response.text)
    #     response = make_response(html_output)
    #     response.headers['Content-Type'] = 'text/html'
    #     return response

    if json_format:
        cleaned_json_str = re.sub(r'```json\n|\n```', '', response.text).strip()
        json_data = json.loads(cleaned_json_str)  # Convert string to JSON (Python object)
        return jsonify({"data": json_data}), 200
    else:
        return jsonify({"data": response.text}), 200


@app.route("/api/v1/writing-test", methods=["POST"])
def test_writer():
    data = request.get_json()

    # Retrieve agentName from query parameters; default to 'default' if not provided
    instruction = data.get('instruction')
    task = get_task("c7845b09d6a64658")
    # Construct the prompt for the chat API
    # user_request = sdf
    final_prompt =   task["prompt"] + "\n" + "instruction:" + instruction

    response = chat_model.generate_content(final_prompt)
    # return response.text
    # Send the prompt to the chat API (dummy implementation here)
    # response = chat.send_message(user_request)
    # return "hay"
    # Return the generated text in JSON format
    cleaned_json_str = re.sub(r'```json\n|\n```', '', response.text).strip()
    json_data = json.loads(cleaned_json_str)  # Convert string to JSON (Python object)
    return jsonify({"data": json_data}), 200

if __name__ == '__main__':
    app.run(debug=True)
