from flask import Blueprint, request, jsonify, make_response
from string import Formatter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from extensions import mongo
from bson import ObjectId
from helpers.ConvertFormat import text_to_html
import google.generativeai as genai
import os
import re
import json

generate_blog_bp = Blueprint('generate_blog', __name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel("gemini-1.5-flash")

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
        {"instruction": 1, "prompt": 1, "is_writer_need": 1, "default_writer": 1, "parameters": 1}
    )

    if task:
        # Convert ObjectId fields to plain strings.
        task["_id"] = str(task["_id"])
        if "default_writer" in task and task["default_writer"]:
            task["default_writer"] = str(task["default_writer"])

    return task


@generate_blog_bp.route('/v1/prompt', methods=['POST'])
def test_content():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    json_format = data.get('json_format')

    if api_key != os.getenv("SYSTEM_API_KEY"):  # need to change
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


@generate_blog_bp.route('/v2/prompt', methods=['POST'])
def prompt_v2():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    json_format = data.get('json_format')
    return_html = data.get('return_html')

    if api_key != os.getenv("SYSTEM_API_KEY"):  # need to change
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

@generate_blog_bp.route('/v1/formatter', methods=['POST'])
def prompt_formatter():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()

    if api_key != os.getenv("SYSTEM_API_KEY"):  # need to change
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


@generate_blog_bp.route('/v2/prompt/editor', methods=['POST'])
def prompt_editor():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    json_format = data.get('json_format')
    return_html = data.get('return_html')

    if api_key != os.getenv("SYSTEM_API_KEY"):  # need to change
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