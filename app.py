from flask import Flask, Blueprint, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from extensions import mongo
from config import Config
import redis
import json
import os

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

mongo.init_app(app)

from routes.writers import seo_writer_bp
from routes.projects import project_bp
from routes.agents import agent_bp
from routes.chatbot import chatbot_bp
from routes.generate_blog import generate_blog_bp
from routes.unstructure_upload import un_structure_bp

API_PREFIX = '/api'
app.register_blueprint(seo_writer_bp, url_prefix=API_PREFIX)
app.register_blueprint(project_bp, url_prefix=API_PREFIX)
app.register_blueprint(agent_bp, url_prefix=API_PREFIX)
app.register_blueprint(chatbot_bp, url_prefix=API_PREFIX)
app.register_blueprint(generate_blog_bp, url_prefix=API_PREFIX)
app.register_blueprint(un_structure_bp, url_prefix=API_PREFIX)

redis_client = redis.Redis(
    host=os.getenv("REDIS_URL"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    username=os.getenv("REDIS_USER_NAME"),
    password=os.getenv("REDIS_API_PASSWORD")
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask MongoDB App!"})

def read_pdf_with_loader(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pdf_text = " ".join([doc.page_content for doc in documents])
    return pdf_text.strip()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    pdf_text = read_pdf_with_loader(filepath)
    if not pdf_text.strip():
        return jsonify({'error': 'The PDF file appears to be empty or could not be read.'}), 400

    loader = PyPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(docs)
    print(docs_split)
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs_split,
        embedding=embeddings,
        index_name="text-embedding-3-small"
    )

    return jsonify({'message': 'PDF embedded successfully', 'chunks': len(docs_split)}), 200

def get_vectorstore():
    embeddings = OpenAIEmbeddings()  # Ensure your OPENAI_API_KEY is in env
    return PineconeVectorStore(
        index_name="text-embedding-3-small",  # Replace with your index name if different
        embedding=embeddings
    )

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
def get_session_flags(session_id):
    """Retrieve session flags like 'has_asked_update' or 'update_declined' from Redis."""
    flags = redis_client.get(f"flags:{session_id}")
    if flags:
        try:
            return json.loads(flags)
        except Exception as e:
            print("Error loading flags:", e)
    return {}


def save_session_flags(session_id, flags_dict):
    """Save session flags in Redis."""
    redis_client.set(f"flags:{session_id}", json.dumps(flags_dict))

@app.route("/api/v2/ai-chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message")
    session_id = data.get("session_id", "default_session")
    name = data.get("name")
    user_input = query

    if not query:
        return jsonify({"error": "Message is required"}), 400

    # Step 1: Get vector store & documents
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    conversation_history = get_memory(session_id)  # returns list of messages
    flags = get_session_flags(session_id)
    has_asked_update = flags.get("has_asked_update", False)
    update_declined = flags.get("update_declined", False)

    if query.strip().lower() in ["no", "nope", "not now", "nah"]:
        update_declined = True
        flags["update_declined"] = True

    update_prompt = ""
    if not has_asked_update and not update_declined:
        update_prompt = "\n\nWould you like to update the instruction based on this?"
        flags["has_asked_update"] = True

    system_prompt = {
        "role": "system",
        "content": f"""You are a smart document assistant. The user has uploaded a document, and relevant content from that document will be retrieved and provided to you when they ask questions.
    
                    Here's how you should behave:
                    
                    - When the user asks a question (e.g., "What does the introduction say?"), use the provided document content to answer accurately.
                    - After answering, ask: "Would you like to update the instruction based on this?" only once per user request.
                    - If the user says "no", do not ask again unless they say they changed their mind.
                    - If the user responds with an update request — even inside a longer message — treat that as confirmation.
                    - As soon as you detect that the user is describing a change they want to make in the document, respond with **only** the word: `update`.
                    - Do not explain or ask again. Just say: `update`, and wait for the system to handle it.
                    
                    Be clear, helpful, and natural. Use only the document content provided. Do not assume anything beyond what's given.
                    
                    Document Context:
                    
                    {context}
                    
                    {update_prompt}"""
    }

    messages = [system_prompt] + conversation_history + [
        {"role": "user", "content": query}
    ]

    response = model.invoke(messages, temperature=0.0)

    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response.content})
    save_memory(session_id, conversation_history)
    save_session_flags(session_id, flags)

    return jsonify({
        "response": response.content
    })


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)