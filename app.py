from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pickle
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Initialize Flask app
app = Flask(__name__)

# Initialize HuggingFaceBgeEmbeddings
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

save_path = "document_embeddings.pkl"

# Load saved embeddings
with open(save_path, "rb") as f:
    saved_data = pickle.load(f)

# Extract documents and embeddings
documents = saved_data["documents"]
document_embeddings = np.array(saved_data["embeddings"])

# FAISS index creation
dimension = document_embeddings.shape[1]  # Number of features in each embedding
index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean distance)

# Add embeddings to FAISS index
index.add(document_embeddings)

# Load the question answering model & tokenizer
# you have to download and save it first from vector_creator.ipynb if you already havent download it locally for offline usage
model = AutoModelForQuestionAnswering.from_pretrained('./roberta-base-squad2')
tokenizer = AutoTokenizer.from_pretrained('./roberta-base-squad2')

# List to store the chat history
chat_history = []

@app.route("/")
def home():
    return render_template("index.html", chat_history=chat_history)

@app.route("/query", methods=["POST"])
def query():
    # Get the question from the user
    user_query = request.json.get("question", "")

    # Generate embedding for the query
    query_embedding = huggingface_embeddings.embed_query(user_query)

    # Search for the most relevant documents in the FAISS index
    k = 3  # Retrieve top 3 most relevant documents
    distances, indices = index.search(np.array([query_embedding]), k)

    # Combine the top 3 documents' contexts
    relevant_docs = [documents[i] for i in indices[0]]
    combined_context = " ".join(relevant_docs)  # Concatenate the contexts for have a full grasp on relevent documents

    # Use the pipeline for question answering
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Pass the question and combined context to the pipeline
    QA_input = {
        'question': user_query,
        'context': combined_context
    }

    # Run the pipeline
    res = nlp(QA_input)

    # Get the answer from the pipeline response
    answer = res['answer']

    # Add the question and answer to chat history
    chat_history.append({
        "question": user_query,
        "answer": answer,
        "relevant_documents": relevant_docs  # Added relevant docs for reference but currently not showing in the chat (optional)
    })

    # Return the result in JSON format, including relevant documents
    return jsonify({
        "answer": answer,
        "relevant_documents": relevant_docs,
        "chat_history": chat_history
    })

if __name__ == "__main__":
    app.run(debug=True)