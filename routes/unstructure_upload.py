import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from flask import Flask, request, jsonify, make_response, Blueprint
from flask_cors import CORS
from flask_pymongo import PyMongo
from datetime import datetime, UTC
from werkzeug.utils import secure_filename
from extensions import mongo


un_structure_bp = Blueprint('un_structure_upload', __name__)

load_dotenv()
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configuration Parameters
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# Initialize the model
model = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)
# Function to clean Markdown formatting from the response
def clean_markdown(text: str) -> str:
    text = re.sub(r'[#*]+', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    return text
# Function to chat with OpenAI using the LangChain model
def chat_with_openai(model, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are an intelligent document parser."},
        {"role": "user", "content": prompt}
    ]
    response = model.invoke(messages)
    return clean_markdown(response.content.strip())
# Function to read PDF using PyPDFLoader
def read_pdf_with_loader(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pdf_text = " ".join([doc.page_content for doc in documents])
    return pdf_text.strip()
# Function to write JSON data to a file
def export_json_to_file(data: Dict, file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"JSON data exported to {file_path}")
# Function to split text into chunks using LangChain
def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Step 1: Summarize the text to create a better prompt
def summarize_text(chunks: List[str], model) -> str:
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following text into a concise, high-quality prompt:\n{chunk}"
        response = chat_with_openai(model, prompt)
        if response:
            summaries.append(response)
    return " ".join(summaries).strip()
# Step 2: Extract relevant data based on the summary
def extract_from_summary(summary: str, model) -> Dict[str, str]:
    prompt = f"""
    From the following summary, extract the relevant data and provide a response in **valid JSON format** with keys:
    'target_audience', 'tone_voice', 'vocabulary_guidelines', and 'important_notes'.
    For 'tone_voice', identify the PRIMARY tone consistently used throughout the document. Do not list multiple conflicting tones. Instead, create a structured object with:
    - 'primary_tone': The dominant tone of the entire document (e.g., "professional", "friendly", "academic")
    - 'tone_variations': An object where keys are specific contexts (e.g., "customer_communications", "internal_documents", "marketing_materials") and values describe the appropriate tone for each context
    - 'tone_characteristics': An array of specific characteristics that define the tone (e.g., "warm", "authoritative", "concise")
    The 'vocabulary_guidelines' should include:
    - A 'replace' object containing word/phrase replacements as key-value pairs, where the key is the word/phrase to avoid and the value is the preferred alternative.
    - A 'suggested_phrases' array containing recommended vocabulary that aligns with the desired tone and voice.
    - A 'avoid_words' array listing specific words, phrases, or terminology that should be avoided.
    - A 'context_specific_terms' object where keys are specific contexts (e.g., 'academic', 'marketing', 'technical') and values are arrays of terms recommended for those contexts.
    - A 'capitalization_rules' object containing specific terms and their proper capitalization format.
    For each item in 'vocabulary_guidelines', include a brief 'reason' or 'explanation' field when possible to explain why certain terms are preferred or should be avoided.
    For 'important_notes', extract any guidelines, rules, or important instructions about brand voice, writing style,
    formatting requirements, or communication standards that are SPECIFICALLY MENTIONED in the summary. Include
    specific writing conventions, capitalization rules, naming conventions, or usage requirements that appear in the text.
    Do NOT use the examples below in your response unless they genuinely appear in the summary.
    Summary:
    {summary}
    """
    response = chat_with_openai(model, prompt)
    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("The model did not return a valid JSON format. Here is the raw response received:")
        print(response)
        return {"response": cleaned_response}
# Step 3: Furnish and finalize the response
def furnish_json_response(extracted_data: Dict[str, str], model) -> Dict[str, str]:
    prompt = f"""
    Refine the following extracted data and return it as a properly structured JSON object.
    The output must be in pure JSON format, without any markdown or extra text.
    If 'important_notes' is present, format it as an array of string guidelines.
    For 'tone_voice', ensure it has the following structure:
    - 'primary_tone': The dominant tone of the entire document
    - 'tone_variations': An object mapping contexts to appropriate tones
    - 'tone_characteristics': An array of specific characteristics that define the tone
    If the original 'tone_voice' is a simple string, convert it to this structure with the string value as the 'primary_tone'.
    For 'vocabulary_guidelines', ensure it has the following structure:
    - 'replace': an object of word/phrase replacements as key-value pairs
    - 'suggested_phrases': an array of recommended vocabulary terms
    - 'avoid_words': an array of words/phrases to avoid
    - 'context_specific_terms': an object where keys are contexts and values are arrays of recommended terms
    - 'capitalization_rules': an object of terms with their proper capitalization format
    For each item in these categories, include a brief 'reason' field explaining why the term is recommended or should be avoided.
    If any section is missing from the original data but information exists in the extracted data to populate it, please do so.
    Extracted Data:
    {json.dumps(extracted_data, indent=4)}
    """
    response = chat_with_openai(model, prompt)
    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("The model did not return a valid JSON format. Here is the raw response received:")
        print(response)
        return {"response": cleaned_response}

@app.route("/api/v1/get-projects", methods=["GET"])
def get_projects():
    try:
        projects = mongo.db.projects.find({}, {"_id": 1, "projectName": 1})
        project_list = list(projects)
        for t in project_list:
            t["_id"] = str(t["_id"])
        return jsonify(project_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/v1/summarizing", methods=["POST"])
def docs_summarizing():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    file_path = None
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process the PDF using the new functions
        pdf_text = read_pdf_with_loader(file_path)
        if not pdf_text.strip():
            return jsonify({'error': 'The PDF file appears to be empty or could not be read.'}), 400
        # Split text into chunks
        chunks = split_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)
        # Summarize the text
        summary = summarize_text(chunks, model)
        # Extract relevant data based on the summary
        extracted_data = extract_from_summary(summary, model)
        # Furnish and finalize the response
        final_json = furnish_json_response(extracted_data, model)
        # Export the final JSON data to a file
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_structure.json")
        export_json_to_file(final_json, output_path)
        return jsonify({
            'status': 'success',
            'summary': summary,
            'extracted_data': extracted_data,
            'final_json': final_json,
            'output_file': output_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
@app.route("/api/v1/create-summarizing", methods=["POST"])
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
            "summary_data": summary_data,
            "created_at": now,
            "updated_at": now
        }
        result = mongo.db.agent_writers.insert_one(document)
        return {"inserted_id": str(result.inserted_id), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}