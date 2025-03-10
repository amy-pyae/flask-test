from flask import Flask,jsonify
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("API_KEY")

@app.route('/')
def hello_world():
    return jsonify({"message": "test message", "key": api_key}), 200
