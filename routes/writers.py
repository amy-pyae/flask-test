from flask import Blueprint, request, jsonify
from extensions import mongo
from datetime import datetime, UTC
from bson import ObjectId
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader
from helpers.FileReader import extract_exact_fields_from_excel
import json
import os
import docx2txt

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

seo_writer_bp = Blueprint('seo_writer', __name__)

@seo_writer_bp.route('/v2/persona/all', methods=['GET'])
def get_personal_all():
    personas = list(mongo.db.agent_writers.aggregate([
        {
            "$addFields": {
                "project_id_obj": { "$toObjectId": "$project_id" },
                "created_by_obj": { "$toObjectId": "$created_by" }
            }
        },
        {
            "$lookup": {
                "from": "projects",
                "localField": "project_id_obj",
                "foreignField": "_id",
                "as": "project"
            }
        },
        {
            "$lookup": {
                "from": "users",
                "localField": "created_by_obj",
                "foreignField": "_id",
                "as": "creator"
            }
        },
        {
            "$unwind": { "path": "$project", "preserveNullAndEmptyArrays": True }
        },
        {
            "$unwind": { "path": "$creator", "preserveNullAndEmptyArrays": True }
        },
        {
            "$project": {
                "name": 1,
                "project_name": "$project.projectName",
                "created_by": "$creator.displayName"
            }
        }
    ]))

    for t in personas:
        t["_id"] = str(t["_id"])

    return jsonify(personas)

@seo_writer_bp.route('/v1/persona/all', methods=['GET'])
def get_personal_all_v2():
    personas = list(mongo.db.agent_writers.aggregate([
        {"$project": {
            "name": 1
        }}
    ]))
    for t in personas:
        t["_id"] = str(t["_id"])
        t["industry"] = ""
    return jsonify(personas)

@seo_writer_bp.route('/v1/persona/<id>', methods=['GET'])
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

@seo_writer_bp.route('/v2/persona/<id>', methods=['PUT'])
def update_agent_writer(id):
    data = request.get_json()
    # Validate required fields
    required_fields = ['name', 'summary_data']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    now = datetime.now(UTC)

    # Update the document in the agent_writers collection
    result = mongo.db.agent_writers.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "name": data['name'],
            "summary_data": data['summary_data'],
            "updated_at": now
        }}
    )

    if result.matched_count == 0:
        return jsonify({"error": "SEO Writer not found"}), 404

    return jsonify({"id": id}), 200

@seo_writer_bp.route('/v2/persona/deploy/<id>', methods=['post'])
def deploy_writer(id):
    data = request.get_json()

    # Validate required fields
    required_fields = ['name', 'summary_data']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    now = datetime.now(UTC)

    result = mongo.db.agent_writers.update_one(
        { "_id": ObjectId(id) },
        {"$set": {
            "name": data['name'],
            "summary_data": data['summary_data'],
            "updated_at": now,
            "deploy_by": data['user_id']
        }}
    )

    if result.matched_count == 0:
        return jsonify({"error": "SEO Writer not found"}), 404

    return jsonify({"id": id}), 200


@seo_writer_bp.route("/v2/persona/create", methods=["POST"])
def create_summary_content():
    data = request.json
    user_id = data.get("user_id")
    project_id = data.get("project_id")
    name = data.get("name")
    summary_data = data.get("summary_data")
    now = datetime.now(UTC)

    try:
        document = {
            "name": name,
            "project_id": project_id,
            "created_by": user_id,
            "prompt":"",
            "instruction":"",
            "summary_data":summary_data,
            "created_at": now,
            "updated_at": now
        }
        print(document)
        result = mongo.db.agent_writers.insert_one(document)
        return {"inserted_id": str(result.inserted_id), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@seo_writer_bp.route("/v2/persona/structured-data/summarizing", methods=["POST"])
def docs_summarizing():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext = filename.rsplit('.', 1)[1].lower()
    try:
        if ext == 'pdf':
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            text = " ".join([doc.page_content for doc in docs])
        elif ext == 'docx':
            text = docx2txt.process(filepath)
        elif ext == 'xlsx':
            summary_data = extract_exact_fields_from_excel(filepath)

            text = json.dumps(summary_data)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400


        return {"summarize": json.loads(text), "status": "success"}

    except Exception as e:
        return jsonify({'error': str(e)}), 500