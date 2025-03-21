from flask import Flask, request, jsonify,make_response
from flask_cors import CORS
from flask_pymongo import PyMongo
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnableBranch
import google.generativeai as genai
from bson import ObjectId
from dotenv import load_dotenv
from string import Formatter
import json
import redis
import os
import hashlib
import jwt
import re
import json
from datetime import datetime, UTC

load_dotenv()

app = Flask(__name__)
CORS(app)


# Configure
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
api_key = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini")
mongo = PyMongo(app)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel("gemini-1.5-flash")
redis_client = redis.Redis(
    host=os.getenv("REDIS_URL"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    username=os.getenv("REDIS_USER_NAME"),
    password=os.getenv("REDIS_API_PASSWORD")
)


# Initialize Gemini Pro Chat Model
chat_model = genai.GenerativeModel("gemini-1.5-flash")
redis_client = redis.Redis(
    host=os.getenv("REDIS_URL"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    username=os.getenv("REDIS_USER_NAME"),
    password=os.getenv("REDIS_API_PASSWORD")
)

def generate_key(data: str):
    return hashlib.sha256(data.encode()).hexdigest()[:16]  # Fixed 16 chars

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask MongoDB App v-0.1"})


@app.route('/api/v1/persona/create', methods=['POST'])
def create_agent_writer():
    data = request.get_json()
    # Validate required fields
    required_fields = ['name','description', 'instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    now = datetime.now(UTC)

    # Insert the new document into the agent_writers collection
    result = mongo.db.agent_writers.insert_one({
        "name": data['name'],
        "description": data['description'],
        "instruction": data['instruction'],
        "prompt": data['instruction'],
        "created_by": None,
        "created_at": now
    })
    return jsonify({
                        "id": str(result.inserted_id),
                        "data": str(result)
                    }), 201

@app.route('/api/v1/persona/<id>', methods=['PUT'])
def update_agent_writer(id):
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'description', 'instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    now = datetime.now(UTC)

    # Update the document in the agent_writers collection
    result = mongo.db.agent_writers.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "name": data['name'],
            "description": data['description'],
            "instruction": data['instruction'],
            "updated_at": now
        }}
    )

    if result.matched_count == 0:
        return jsonify({"error": "Persona not found"}), 404

    return jsonify({"id": id}), 200

@app.route('/api/v1/persona/deploy/<id>', methods=['post'])
def deploy_writer(id):
    data = request.get_json()

    # Validate required fields
    required_fields = ['instruction']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    now = datetime.now(UTC)

    # Update the document in the agent_writers collection
    result = mongo.db.agent_writers.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "name": data['name'],
            "description": data['description'],
            "instruction": data['instruction'],
            "prompt": data['instruction'],
            "updated_at": now,
            "deploy_by": None
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
            "instruction": 1,
            "description": 1
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
    return_html = data.get('return_html')


    if api_key != os.getenv("SYSTEM_API_KEY"): # need to change
        return jsonify({"error": "Invalid api key"}), 400


    task_key = data.get('key', '').strip()
    parameters = data.get('parameters', {})
    # print(parameters['content'])
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

    if return_html:
        html_output = text_to_html(response.text)
        response = make_response(html_output)
        response.headers['Content-Type'] = 'text/html'
        return response

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

@app.route('/api/v1/formatter', methods=['POST'])
def prompt_formatter():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()

    if api_key != os.getenv("SYSTEM_API_KEY"): # need to change
        return jsonify({"error": "Invalid api key"}), 400

    task_key = data.get('key', '').strip()
    parameters = data.get('parameters', {})

    task = get_task(task_key)

    final_prompt = task["prompt"]
    prompt = task["prompt"]

    final_prompt = prompt.replace("<<<CONTENT>>>", parameters["content"])

    try:
        response = chat_model.generate_content(final_prompt)
    except Exception as e:
            return jsonify({"error": f"Error calling Gemini API: {str(e)}"}), 500
    #
    #
    #
    cleaned_json_str = re.sub(r'```json\n|\n```', '', response.text).strip()
    json_data = json.loads(cleaned_json_str)  # Convert string to JSON (Python object)
    return jsonify({"data": json_data}), 200

@app.route('/api/v2/prompt/editor', methods=['POST'])
def prompt_editor():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    json_format = data.get('json_format')
    return_html = data.get('return_html')

    if api_key != os.getenv("SYSTEM_API_KEY"): # need to change
        return jsonify({"error": "Invalid api key"}), 400


    # task_key = data.get('key', '').strip()
    parameters = data.get('parameters', {})
    instruction = parameters['user_input']
    original_content = parameters['orginal_content']
    target_content = parameters['target_content']
    llm = data.get('llm', {})

    # task = get_task(task_key)

    # if not task:
    #     return jsonify({"error": "Task not found"}), 404

    final_prompt = f"""You are an AI content editor. Modify **only** the Target Content according to the User Instruction, keeping the Original Content unchanged. Return the full text with the modified section seamlessly integrated.

                        User Instruction:
                        {instruction}
                    
                        Target Content (Selected by User) (Modify this text only):
                        {target_content}
                    
                        Original Content (For Reference, Will Be Returned):
                        {original_content}
                    
                        Guidelines:
                        - Modify **only** the Target Content based on the User Instruction.
                        - Do **not** alter the rest of the Original Content.
                        - Ensure the modified section integrates naturally into the full text.
                        - Maintain readability, coherence, and original meaning unless otherwise specified.
    """

    try:
        response = chat_model.generate_content(final_prompt)
    except Exception as e:
            return jsonify({"error": f"Error calling Gemini API: {str(e)}"}), 500

    if return_html:
        html_output = text_to_html(response.text)
        response = make_response(html_output)
        response.headers['Content-Type'] = 'text/html'
        return response

    if json_format:
        cleaned_json_str = re.sub(r'```json\n|\n```', '', response.text).strip()
        json_data = json.loads(cleaned_json_str)  # Convert string to JSON (Python object)
        return jsonify({"data": json_data}), 200
    else:
        return jsonify({"data": response.text}), 200

#===================================
def get_memory(session_id):
    """Retrieve the summarized conversation from Redis as a list."""
    summary = redis_client.get(f"summary:{session_id}")
    if summary:
        try:
            return json.loads(summary)
        except Exception as e:
            print("Error loading memory:", e)
    return []


def save_memory(session_id, conversation_list):
    """Store the updated conversation list in Redis."""
    redis_client.set(f"summary:{session_id}", json.dumps(conversation_list))


@app.route("/api/v1/check-instruction", methods=["POST"])
def is_instruction_request():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")

    # conversation_history = get_memory(session_id)

    messages = [
        {"role": "system", "content": f""" You are an AI assistant analyzing a conversation where a user is designing or modifying an AI assistant.

                                        Determine if the latest message shows that the user is shaping, refining, or modifying the assistant's behavior, purpose, or instructions.

                                        Latest User Message:
                                        "{user_input}"

                                        Respond with:
                                        - yes — if the user is creating, modifying, asking about, or clarifying how the assistant should behave or what it should do/avoid.
                                        - no — if the message is unrelated to defining the assistant's function (e.g., general questions, small talk, unrelated tasks).

                                        Only reply with "yes" or "no". Do not include quotes or additional text."""
         }
    ]

    response = model.invoke(messages, temperature=0)
    classification = response.content.strip().lower()
    return jsonify({
        "response": classification == "yes"
    })  # classification = response.content.strip().lower()
    #
    # return classification == "yes"


@app.route("/api/v1/update-instruction", methods=["POST"])
def update_instruction():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    existing_instruction = data.get("existing_instruction")
    new_info = data.get("new_instruction")

    # conversation_history = get_memory(session_id)

    # print(new_info)

    prompt = f"""
    Update the following assistant description to naturally include the instruction: "{user_input}"

    Keep the original structure, tone, and formatting. Only modify where necessary to include the new instruction smoothly.

    Return only the updated description, with no extra explanation or formatting.

    Assistant description:

    \"\"\"{existing_instruction}\"\"\"
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    update_response = model.invoke(messages, temperature=0)
    print(update_response)

    # update_response = model.invoke(messages_update, temperature=0)
    updated_instruction = update_response.content.strip('""""')
    return updated_instruction


@app.route("/api/v1/get-instruction", methods=["POST"])
def get_ai_generated_instruction():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    # 🔹 Retrieve conversation history from Redis
    # conversation_history = get_memory(session_id)

    new_instruction = generate_ai_system_prompt(user_input)

    ai_suggested_name = suggest_profile_name(new_instruction)

    return jsonify({
        "new_instruction": new_instruction,
        "ai_suggested_description": ai_suggested_name
    })


def suggest_profile_name(instruction):
    """Suggests a meaningful assistant name based on conversation context."""
    messages = [
        {"role": "system", "content": """
            Generate a **concise, relevant, and descriptive** phrase summarizing the AI assistant's purpose. 

            **Guidelines:**
            - Extract the **core function** from the given instruction.
            - Keep the description **short (less than 15 words)**.
            - Ensure it sounds **natural and professional**.
            - Avoid generic terms like "AI Assistant" and instead describe **what the assistant does**.

            **Example Outputs:**
            - Input: "Make a creative writing assistant"
              Output: "A creative writing assistant for brainstorming and refining stories."
            - Input: "Make a software engineer assistant"
              Output: "A software engineer assistant for formatting and refining code."

            Now, generate a short description for the following instruction.
        """},
        {"role": "user", "content": str(instruction)}
    ]

    response = model.invoke(messages, temperature=0.5)

    try:
        extracted_data = json.loads(response.content.strip())  # Ensure valid JSON
        return extracted_data
    except Exception as e:
        print(f"Error suggesting assistant name: {e}")
        return "AI Assistant"  # Fallback name


def generate_ai_system_prompt(user_prompt):
    messages = [
        {"role": "system",
         "content": "You are an AI that generates concise, structured, and natural-sounding instructions for a Custom GPT."},
        {"role": "system", "content": f"""
            Generate a **short, structured, and engaging** instruction for a GPT assistant in a natural tone.

            **Format the response as follows:**
            - Begin with: **"This OOm Writing Assistant is a..."** followed by a **clear, direct description** of what it does.
            - Follow with **a sentence or two** explaining how it interacts with users in a supportive and engaging way.
            - End with a **short, smooth statement** about what it avoids, ensuring flexibility while setting constraints.

            The response should be **brief, expressive, and fluid**—avoiding section titles, lists, or rigid formatting.

            Now, generate an instruction for a GPT that serves as a {user_prompt}.
        """},
    ]

    try:
        response = model.invoke(messages, temperature=0.7)
        refined_instruction = response.content.strip()
        return refined_instruction

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error generating refined instruction."


def update_title(updated_instruction):
    title_messages = [
        {"role": "system", "content": """
                                            Analyze the following AI assistant description and extract a **concise assistant profile name**.

                                            **Requirements:**
                                            - The name should be **clear, relevant, and meaningful**.
                                            - **Do NOT exceed 50 characters**.
                                            - Keep it professional and structured.
                                            - Do NOT include unnecessary words like "This GPT is...".
                                            - Return ONLY a valid JSON response without any extra text or formatting.

                                            Output format:
                                            {
                                                "assistantProfile": ""
                                            }
                                        """
         },
        {"role": "user", "content": updated_instruction}
    ]
    title_response = model.invoke(title_messages, temperature=0)
    extracted_title = title_response.content.strip()

    return extracted_title


def safe_openai_call(messages, temperature=0.7):
    """Safely invoke OpenAI API with error handling."""
    try:
        response = model.invoke(messages, temperature=temperature)
        return response.content.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "I'm experiencing technical issues. Please try again later."


def extract_assistant_name(conversation_history):
    """Extract a meaningful assistant name based on previous interactions."""
    messages = [
        {"role": "system", "content": """
            Analyze the following conversation and determine the **best possible name** for the AI assistant.

            **Guidelines:**
            - The name should be **clear, relevant, and professional**.
            - **Do NOT exceed 50 characters**.
            - Avoid generic names like "AI" or "Assistant" alone.
            - Keep it simple yet descriptive (e.g., "AI Writing Assistant").
            - Return only a **valid JSON object**.

            **Output format:**
            { "assistantProfile": "Assistant Name Here" }
        """},
        {"role": "user", "content": str(conversation_history)}
    ]
    response = model.invoke(messages, temperature=0)

    try:
        extracted_data = json.loads(response.content.strip())  # Ensure valid JSON
        return extracted_data.get("assistantProfile", "AI Assistant")  # Default fallback
    except Exception as e:
        print(f"Error extracting assistant name: {e}")
        return "AI Assistant"  # Fallback name


def update_conversation_history(session_id, user_input, ai_response):
    """Update conversation history and prevent duplicates."""
    conversation_history = get_memory(session_id)

    if conversation_history:
        last_entry = conversation_history[-1]
        if last_entry.get("role") == "assistant" and last_entry.get("content") == ai_response:
            return  # Avoid duplicate AI responses

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": ai_response})

    save_memory(session_id, conversation_history)


# Using
def extract_custom_name(user_input):
    messages = [
        {"role": "system", "content": """ 
            Extract the assistant name from the user's message.

            **Instructions:**
            - The user might provide a name directly, such as:
              - "I prefer BrainyBot."
              - "Let’s call it AI Master."
              - "How about Pro Max Specialist?"
            - Only return the **extracted name** without any extra text.

            **Example Outputs:**
            - **User:** "I prefer BrainyBot." → **Response:** `"BrainyBot"`
            - **User:** "Let’s name it Pro Max Specialist." → **Response:** `"Pro Max Specialist"`
            - **User:** "I like AI Genius." → **Response:** `"AI Genius"`

            Do NOT include extra characters, explanations, or formatting or "" or single quote
        """},
        {"role": "user", "content": user_input}
    ]

    response = model.invoke(messages, temperature=0)
    return response.content.strip()


# Using
def suggest_assistant_name(conversation_history):
    """Suggests a meaningful assistant name based on conversation context."""
    messages = [
        {"role": "system", "content": """
            Analyze the following conversation and suggest the **best possible name** for the AI assistant.

            **Guidelines:**
            - The name should be **clear, relevant, and professional**.
            - **Do NOT exceed 50 characters**.
            - Avoid generic names like "AI" or "Assistant" alone.
            - Suggest something meaningful based on the context.
            - Keep it simple yet descriptive (e.g., "AI Writing Assistant").
            - Return only a **valid JSON object**.

            **Output format:**
            { "suggestedName": "Assistant Name Here" }
        """},
        {"role": "user", "content": str(conversation_history)}
    ]

    response = model.invoke(messages, temperature=0)

    try:
        extracted_data = json.loads(response.content.strip())  # Ensure valid JSON
        return extracted_data.get("suggestedName", "AI Assistant")  # Default fallback
    except Exception as e:
        print(f"Error suggesting assistant name: {e}")
        return "AI Assistant"  # Fallback name


# Using
def detect_name_confirmation(conversation_history, user_input):
    print(conversation_history)
    messages = [
        {"role": "system", "content": """ 
                You are analyzing user responses to determine if they are confirming, rejecting, or changing the topic.

                **Instructions:**
                - If the user confirms (e.g., "That works!", "Sounds great", "I like it"), respond with `"confirmed"`.
                - If the user rejects and wants a new name (e.g., "No", "Nah, another one", "I don’t like that"), respond with `"rejected"`.
                - If the user suggests their own name (e.g., "I prefer BrainyBot", "Let’s call it AI Master"), respond with `"custom_name"`.
                - If the user has changed the topic (e.g., "Tell me about SEO", "How do I use AI for writing?"), respond with `"topic_change"`.

                **Response format:**
                Return only one of the following exact words:
                - `"confirmed"`
                - `"rejected"`
                - `"custom_name"`
                - `"topic_change"`

                Do NOT include extra characters, explanations, or formatting.
            """}
    ]

    # Include old conversation history to provide context
    if conversation_history:
        messages.extend(conversation_history)

    # Add the latest user message
    messages.append({"role": "user", "content": user_input})
    response = model.invoke(messages, temperature=0.1)

    return response.content.strip()


@app.route("/api/v1/test-instruction", methods=["POST"])
def test_instruction():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    name = data.get("name")
    description = data.get("description")
    conversation_history = get_memory(session_id)

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    # messages = [{"role": "user", "content": user_input}]
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant being configured by the user."),
        SystemMessage(content="Your goal is to update the assistant's behavior naturally based on user requests."),
        SystemMessage(content=f"Current Assistant Configuration: {instruction}"),
        SystemMessage(content=f"Current Assistant Name: {name}"),
        SystemMessage(content=f"Current Assistant Description: {description}"),
        MessagesPlaceholder(variable_name="conversation_history"),
        HumanMessage(content=user_input)
    ])

    response = prompt_template | model
    ai_response = response.invoke(
        {"instruction": instruction, "conversation_history": conversation_history})  # ✅ Apply instruction

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": ai_response.content})
    save_memory(session_id, conversation_history)

    return jsonify({"response": ai_response.content.strip()})


@app.route("/api/v1/input-classification", methods=["POST"])
def input_classification():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    conversation_history = get_memory(session_id)

    messages = [
        {
            "role": "system",
            "content": """
                            You are an AI assistant analyzing a conversation to determine the user's intent.

                            **Your task:** Based on the conversation so far and the user's latest message, classify what the user is trying to do.

                            **Possible classifications:**
                            - name_update: The user wants to rename the assistant or give it a new name.
                            - instruction_update: The user is trying to change, update, or define how the assistant behaves, responds, or what capabilities it has.
                            - chit_chat: The user is engaging in casual talk, greetings, or unrelated small talk.
                            - unclear: The message is vague, incomplete, or doesn't clearly match any of the above categories.

                            **Guidelines:**
                            - Use the latest message to infer intent.
                            - Use prior conversation only if the latest message is ambiguous.
                            - If the user is giving a task, prompt, or desired behavior (e.g., "I want to create...", "Make it respond like...", "It should..."), classify as instruction_update.
                            - Do not interpret or explain. Just classify.

                            **Output format:**
                            Return only one of the classification strings: name_update, instruction_update, chit_chat, or unclear

                            Do NOT include quotes, punctuation, or extra text.
                        """
        }
    ]
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_input})
    response = model.invoke(messages, temperature=0)

    classification = response.content.strip().lower()

    conversation_history.append({"role": "user", "content": user_input})
    # conversation_history.append({"role": "assistant", "content": classification})
    save_memory(session_id, conversation_history)
    return jsonify({
        "response": classification
    })


@app.route("/api/v1/ai-extract-name", methods=["POST"])
def ai_extract_name():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    is_name_confirmation = data.get("is_name_confirmation")
    suggested_name = data.get("suggested_name")
    conversation_history = get_memory(session_id)

    custom_name = extract_custom_name(user_input)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": custom_name})
    save_memory(session_id, conversation_history)

    return jsonify(
        {"response": "custom_name", "detect_name_confirmation": False, "suggested_name": custom_name}), 200


@app.route("/api/v1/ai-name-confirmation", methods=["POST"])
def get_ai_name_confirmation():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    is_name_confirmation = data.get("is_name_confirmation")
    suggested_name = data.get("suggested_name")
    conversation_history = get_memory(session_id)

    if isinstance(is_name_confirmation, str):
        is_name_confirmation = is_name_confirmation.lower() == "true"

    if is_name_confirmation:
        detect_confirmation = detect_name_confirmation(conversation_history, user_input).strip().replace('"',
                                                                                                         '').lower()
        print(detect_confirmation)
        if detect_confirmation == "confirmed":
            return jsonify(
                {"response": "confirmed", "detect_name_confirmation": False, "suggested_name": suggested_name}), 200

        elif detect_confirmation == "rejected":
            return jsonify(
                {"response": "rejected", "detect_name_confirmation": True, "suggested_name": suggested_name}), 200

        elif detect_confirmation == "topic_change":
            return jsonify(
                {"response": "change", "detect_name_confirmation": False, "suggested_name": suggested_name}), 200
        elif detect_confirmation == "custom_name":
            custom_name = extract_custom_name(user_input)
            return jsonify(
                {"response": "custom_name", "detect_name_confirmation": False, "suggested_name": custom_name}), 200

    conversation_history.append({"role": "user", "content": user_input})
    save_memory(session_id, conversation_history)

    return jsonify(
        {"response": "no_confirmation", "detect_name_confirmation": False, "suggested_name": suggested_name}), 200


@app.route("/api/v1/ai-name-suggestion", methods=["POST"])
def get_ai_name_suggestion():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")

    conversation_history = get_memory(session_id)
    print(str(conversation_history))
    messages = [
        {"role": "system", "content": f"""
                   Analyze the following conversation and suggest the **best possible name** for the AI assistant.

                   **Guidelines:**
                   - Describe the assistant type in **3-5 words** (e.g., "AI Writing Assistant", "SEO Optimization Bot").
                   - Suggest a **creative, unique name** based on the assistant's purpose.
                   - **Avoid previously suggested names.
                   - Do NOT repeat past suggestions.
                   - Ensure the name is **concise and meaningful**.

                   **Return only the assistant name, no extra text.**
           """},
        {"role": "user", "content": str(conversation_history)}
    ]

    assistant_type = model.invoke(messages, temperature=0.9).content.strip()

    suggested_name = suggest_assistant_name(user_input)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": assistant_type})
    save_memory(session_id, conversation_history)
    # suggested_name = "Software Engineer"
    # redis_client.set(f"pending_name:{session_id}", suggested_name)

    return jsonify({
        "response": f"""Now, let's give this Assistant a name. How about **{suggested_name}**? Let me know if you have a different preference!""",
        "suggested_name": suggested_name
    })


@app.route("/api/v1/test-chat", methods=["POST"])
def test_chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")
    conversation_history = get_memory(session_id)

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    # # 🔹 Retrieve conversation history from Redis
    # conversation_history = get_memory(session_id)
    #
    # # 🔹 Use a single OpenAI call to classify user input
    # classification_prompt = f"""
    #         The user is interacting with an AI assistant. Determine what they are trying to do.
    #
    #         **User Message:** "{user_input}"
    #
    #         **Classify the intent as one of the following:**
    #         - "name_update": If the user wants to rename the assistant.
    #         - "instruction_update": If the user is modifying how the assistant should behave.
    #         - "chit_chat": If it's a general conversation.
    #         - "unclear": If the intent is unclear.
    #
    #         Return only the classification string.
    # """
    #
    # classification = model.invoke([
    #     {"role": "system", "content": classification_prompt}
    # ], temperature=0).content.strip().lower()
    #
    # print((classification))
    # # 🔹 Handle different classifications with minimal calls
    # if classification == "name_update":
    #     extracted_name = user_input.strip()
    #     return jsonify({
    #         "response": f"Got it! Your assistant is now named **{extracted_name}**.",
    #         "updatedName": extracted_name
    #     })
    #
    # if classification == "instruction_update":
    #     if not instruction:
    #         print('Initial Instruction')
    #         updated_instruction = generate_ai_system_prompt(user_input)  # Generate new instruction
    #     else:
    #         print('Update Instruction')
    #         updated_instruction = update_instruction(instruction, user_input)  # Update instruction with user input
    #
    #     # updated_instruction = update_instruction(instruction or "", user_input)
    #     # extracted_title = update_title(updated_instruction)
    #     return jsonify({
    #         "response": "Your assistant's behavior has been updated.",
    #         "updatedInstruction": updated_instruction,
    #         # "updatedName": extracted_title
    #     })

    messages = [
        # {"role": "system", "content": f"""{info}"""},
        {"role": "system", "content": f"""You are an **Intelligent Writing Assistant** that helps users define their **Writing Style**.

                    ## **Step 1: Ask Key Questions**
                    - Start with **three core questions**:
                      1. **What should the assistant do?**  
                         (Examples: Write concisely, provide detailed explanations, use storytelling, persuasive tone.)
                      2. **What should the assistant avoid?**  
                         (Examples: Jargon, passive voice, overly complex words, unnecessary details.)
                      3. **How should the assistant behave?**  
                         (Examples: Be proactive with suggestions, strictly follow rules, allow creativity and flexibility.)

                    - If the user is unsure, **offer predefined choices** to help them decide.

                    ## **Step 2: Summarize & Confirm**
                    - Summarize their preferences in a structured **Writing Style Guide**"""}
    ]

    response = model.invoke(messages, temperature=0.8)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response.content})
    save_memory(session_id, conversation_history)

    return jsonify({
        "response": response.content
    })


def refine_instruction(user_input):
    # Make a request to OpenAI for improving the instruction
    prompt = f"""
                   You are a system responsible for transforming user requests into well-structured AI assistant descriptions.

    The user has provided the following instruction for their AI assistant:
    "${user_input}"

    **Task:** Rewrite this instruction into a clear, well-structured AI system prompt while preserving its original intent and meaning.

    **Guidelines:**
    - **Ensure the description starts with "This Bot is..."**
    - Ensure the description is professional and coherent.
    - Expand relevant details if necessary to improve clarity.
    - Do **not** add unnecessary explanations or assumptions.
    - Make it structured but not overly robotic.

    **Output format:** Only return the structured instruction text, without additional formatting, examples, or explanations.
                """

    try:
        # Use the 'invoke' method from langchain's ChatOpenAI instead of Completion.create
        response = model.invoke([
            {"role": "system", "content": prompt}
        ], temperature=0.7)

        refined_instruction = response.content.strip()
        return refined_instruction

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error generating refined instruction."


@app.route("/api/v1/set-initial", methods=["POST"])
def set_initial_instruction():
    data = request.json
    instruction = data.get("instruction")
    name = data.get("name")
    description = data.get("description")
    session_id = data.get("session_id", "default_session")
    conversation_history = get_memory(session_id)
    print("Initial Session Id", session_id)

    if description:
        conversation_history.append({"role": "user", "content": f"name: {name}."})
    if name:
        conversation_history.append({"role": "user", "content": f"description: {description}"})
    if instruction:
        conversation_history.append({"role": "user", "content": instruction})

    save_memory(session_id, conversation_history)
    return jsonify({
        "response": "success"
    })


@app.route("/api/v1/ai-chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    conversation_history = get_memory(session_id)

    messages = [
        # {"role": "system", "content": f"""{instruction}"""},
        {"role": "system", "content": f"""
        You are a helpful and friendly assistant designed to guide users as they write or refine instructions for an AI assistant.

        Your role is to support the user step by step, keeping the conversation clear and productive.

        Guidelines:
        - If the user seems unsure or confused, suggest one or two specific next steps in a short, supportive response.
        - If the user's request is vague, ask one clear, focused follow-up question instead of guessing their intent.
        - Keep all responses short, conversational, and action-oriented to help the user move forward.
        - Do not explain concepts unless the user asks for clarification.
        - Always base your reply on the latest user message and the flow of the conversation.
        """}
    ]

    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    response = model.invoke(messages, temperature=0.3)

    # conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response.content})
    save_memory(session_id, conversation_history)

    print("AI Chat", conversation_history)
    return jsonify({
        "response": response.content
    })


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
