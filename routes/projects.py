from flask import Blueprint, request, jsonify
from extensions import mongo

project_bp = Blueprint('project', __name__)

@project_bp.route("/v2/get-projects", methods=["GET"])
def get_projects():
    try:
        projects = mongo.db.projects.find({}, {"_id": 1, "projectName": 1})
        project_list = list(projects)
        for t in project_list:
            t["_id"] = str(t["_id"])
        return jsonify(project_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@project_bp.route("/v1/get-projects", methods=["GET"])
def get_projects_v1():
    try:
        projects = mongo.db.projects.find({}, {"_id": 1, "projectName": 1})
        project_list = list(projects)
        for t in project_list:
            t["_id"] = str(t["_id"])
        return jsonify(project_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500