from flask import Flask, request, render_template, session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path

app = Flask(__name__, template_folder="Template")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "PDF_files"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/",methods=["GET","POST"])
def login():
    return render_template("index2.html")

@app.route("/upload", methods=["POST"])
def upload():
    session["context_question"] = []

    question = (request.form.get("textInput") or "").strip()
    uploaded_file = request.files.get("pdfFile")

    if not question:
        return render_template("index1.html", answer="Please enter a question.")

    if not uploaded_file or not uploaded_file.filename:
        return render_template("index1.html", answer="Please choose a PDF file.")

    if not allowed_file(uploaded_file.filename):
        return render_template("index1.html", answer="Please upload a PDF file.")

    safe_name = secure_filename(uploaded_file.filename)
    if not safe_name:
        return render_template("index1.html", answer="The uploaded filename is invalid.")

    # A unique name prevents uploads with the same original filename from colliding.
    file_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{safe_name}"
    uploaded_file.save(file_path)

    try:
        reader = PdfReader(str(file_path))
        pages = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_number,
                        },
                    )
                )

        if not pages:
            return render_template(
                "index1.html",
                answer="The PDF contains no readable text."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
             )
        docs = text_splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        db = FAISS.from_documents(docs, embeddings)
        search_docs = db.similarity_search(question, k=3)

        if not search_docs:
            return render_template(
                "index1.html",
                answer="I couldn't find relevant information in that PDF."
            )

        context = search_docs[0].page_content
        session["context_question"] = [
            (doc.page_content, question) for doc in search_docs
        ]
        session["feedback_index"] = 0

        model_name = "Intel/dynamic_tinybert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        question_answerer = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=tokenizer,
        )
        result = question_answerer(question=question, context=context)

        return render_template(
            "index1.html",
            answer=f"answer: {result['answer']}",
            context=f"context: {context}",
            question=f"question: {question}",
        )

    except Exception:
        app.logger.exception("Failed to process uploaded PDF")
        return render_template(
            "index1.html",
            answer="I couldn't read or process that PDF. Please try another file."
        )
    finally:
        file_path.unlink(missing_ok=True)
                

@app.route("/upload/feedback", methods=["POST"])
def feedback():
    context_question = session.get("context_question", [])

    if not context_question:
        return render_template(
            "index1.html",
            answer="Please upload a PDF and ask a question first.",
        )

    if request.form.get("btnTask2") == "click2":
        session.pop("context_question", None)
        session.pop("feedback_index", None)
        return render_template(
            "index1.html",
            answer="Thank you for your positive feedback.",
        )

    if request.form.get("btnTask3") == "click3":
        next_index = session.get("feedback_index", 0) + 1

        if next_index >= len(context_question):
            session.pop("context_question", None)
            session.pop("feedback_index", None)
            return render_template(
                "index1.html",
                answer="No more relevant passages were found. Please try another question.",
            )

        session["feedback_index"] = next_index
        context, question = context_question[next_index]

        try:
            model_name = "Intel/dynamic_tinybert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            question_answerer = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=tokenizer,
            )
            result = question_answerer(question=question, context=context)

            return render_template(
                "index1.html",
                answer=f"answer: {result['answer']}",
                context=f"context: {context}",
                question=f"question: {question}",
            )
        except Exception:
            app.logger.exception("Failed to generate an answer from another passage")
            return render_template(
                "index1.html",
                answer="I couldn't generate another answer. Please try again.",
            )

    return render_template(
        "index1.html",
        answer="Please choose a feedback option.",
    )
if __name__ == "__main__":
    app.run(debug=True)