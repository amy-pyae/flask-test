import os
import json
import re
import tempfile
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from flask import Flask, request, jsonify, make_response, Blueprint
from werkzeug.utils import secure_filename


un_structure_bp = Blueprint('un_structure_upload', __name__)

load_dotenv()
UPLOAD_FOLDER = 'uploads'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    From the following summary, **thoroughly extract all relevant data** and provide a response in **valid JSON format** with the keys:
    - **'target_audience'**: Analyze the document to identify the target audience. 
    - **'tone_voice'**: Identify the overall tone used throughout the document. 
        - **'primary_tone'**: Clearly specify the dominant tone that best represents the overall communication style (e.g., "professional", "friendly", "academic"). Do not list multiple conflicting tones; provide a single, cohesive descriptor.
        - **'tone_variations'**: Structure this as an object where keys are specific contexts (e.g., "customer_communications", "internal_documents", "marketing_materials") and values describe the appropriate tone for each context. Ensure no relevant variation is omitted.
        - **'tone_characteristics'**: Provide an array of detailed characteristics that define the tone. Describe how each characteristic is demonstrated in the writing (e.g., "warm", "authoritative", "concise").
    - **'vocabulary_guidelines'**: Extract all vocabulary-related guidance and structure it as follows:
        - **'replace'**: An object containing word/phrase replacements as key-value pairs. The key is the word/phrase to avoid, and the value is the preferred alternative. Include a brief 'reason' or 'explanation' field for each replacement explaining why it should be avoided or preferred.
        - **'suggested_phrases'**: An array of recommended words, phrases, or expressions that align well with the desired tone and voice. Where relevant, explain why they are preferred.
        - **'avoid_words'**: A list of specific words, phrases, or terminology that should be avoided. Include reasons when applicable.
        - **'context_specific_terms'**: An object where keys are contexts (e.g., 'academic', 'marketing', 'technical') and values are arrays of terms recommended for those contexts. Explain why these terms are suitable for their respective contexts if mentioned.
        - **'capitalization_rules'**: An object containing specific terms and their proper capitalization format, along with a brief explanation of the rule if applicable.
    - **'important_notes'**: Extract all guidelines, rules, or instructions related to brand voice, writing style, formatting requirements, or communication standards that are **specifically mentioned**. Include specific writing conventions, capitalization rules, naming conventions, or usage requirements explicitly stated in the text.

    **Important:**  
    - Ensure the JSON output captures all details, including nuanced preferences, terminologies, tone descriptions, and context-specific vocabulary guidance. 
    - Do NOT provide a general summary or interpretation; instead, accurately reflect the precise guidance given in the document.
    - Maintain the original structure and intent of the content while organizing it as structured JSON data.
Summary: [read thru the chunks and make sure you have at least 15-30 words in each vocabulary guidelines category and at least 10 statements in each important notes category]
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
For 'vocabulary_guidelines', ensure it has the following structure:  [must include at least 10 words in each category, read thru the chunks and make sure you have at least 10 words in each category]
- 'replace': an object of word/phrase replacements as key-value pairs  
- 'suggested_phrases': an array of recommended vocabulary terms  
- 'avoid_words': an array of words/phrases to avoid [at least 10 words]
- 'context_specific_terms': an object where keys are contexts and values are arrays of recommended terms  
- 'capitalization_rules': an object of terms with their proper capitalization format
For each item in these categories, include a brief 'reason' field explaining why the term is recommended or should be avoided.  
If any section is missing from the original data but information exists in the extracted data to populate it, please do so.  

Example structure:
{{
    "target_audience": "Parents seeking clarity and understanding about their child's educational journey, and educators who need clear communication.",
    "tone_voice": {{
        "primary_tone": "Sincere and progressive",
        "tone_variations": {{
            "customer_communications": "Gentle and supportive",
            "internal_documents": "Professional and clear"
        }},
        "tone_characteristics": ["warm", "authoritative", "concise"]
    }},
    "vocabulary_guidelines": {{
        "replace": {{
            "Focus": "Maintains engagement in - Encourages deeper involvement and interest",
            "Providing": "Provide opportunity to or for - Suggests a facilitative approach rather than direct action"
        }},
        "suggested_phrases": [
            "Positive disposition",
            "Building a positive connection to friends/school"
        ],
        "avoid_words": [
            "Can't",
            "Don't",
            "Focus"
        ],
        "context_specific_terms": {{
            "academic": ["Observed", "Examined", "Studied"],
            "marketing": ["Positive disposition", "Encourage Joe to..."]
        }},
        "capitalization_rules": {{
            "Eton House": "Use E and H capital"
        }}
    }},
    "important_notes": [
        "Use active voice instead of passive voice.",
        "Use positive language rather than negative language.",
        "Avoid aggressive selling tactics; prefer a knowledgeable and compassionate approach."
    ]
}}

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


@un_structure_bp.route("/v1/un-structure-summarizing", methods=["POST"])
def docs_un_structure_summarizing():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        # Process the PDF using the new functions
        pdf_text = read_pdf_with_loader(temp_path)
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

        return jsonify({
            'status': 'success',
            'chunks': chunks,
            'summary': summary,
            'extracted_data': extracted_data,
            'final_json': final_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
