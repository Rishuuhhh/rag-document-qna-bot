"""
Improved version of the RAG PDF Q&A Flask app.

What changed vs the original app.py
-----------------------------------
1. BUG FIX: the feedback route used `result` before it was defined (crash on "No" feedback).
2. Embedding model + QA model are loaded ONCE (lazy), not on every request -> much faster.
3. All retrieved passages are scored by the QA model and ranked best-first,
   so the "next answer" button really shows the next-best answer.
4. Low-confidence warning (tune LOW_CONFIDENCE after testing in the notebook).
5. Multiple PDFs in one request (works with the old single-file form too).
6. Feedback is logged to feedback_log.csv.
7. Passages are stored server-side (not in the 4 KB cookie session).

Template variables kept the same: answer, context, question, source_page.
Extra variables passed (templates may ignore them): score, source_file.
"""

import csv
import datetime
import os
import uuid
from pathlib import Path

from flask import Flask, request, render_template, session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader
from transformers import AutoTokenizer, pipeline
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "PDF_files"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
FEEDBACK_LOG = BASE_DIR / "feedback_log.csv"

QA_MODEL_NAME = "Intel/dynamic_tinybert"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"
TOP_K = 4               # how many chunks to retrieve from FAISS
LOW_CONFIDENCE = 0.02   # below this, show a "not sure" note (tune it!)
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, template_folder="Template")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB total

# Server-side store: session id -> {"question": str, "items": [...], "index": int}
# (In-memory: fine for a demo / single process. Use Redis/DB for production.)
STORE = {}

# ---------------------------------------------------------------- models (loaded once)
_embeddings = None
_qa = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    return _embeddings


def get_qa():
    global _qa
    if _qa is None:
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        _qa = pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=tokenizer)
    return _qa


# ---------------------------------------------------------------- helpers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_pdfs(files):
    """Read every uploaded PDF page by page, keeping file name + page number."""
    pages = []
    for f in files:
        name = secure_filename(f.filename)
        if not name or not allowed_file(name):
            continue
        path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{name}"
        f.save(path)
        try:
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={"source": name, "page": page_number},
                        )
                    )
        finally:
            path.unlink(missing_ok=True)
    return pages


def rank_passages(question, docs):
    """Run the QA model on every retrieved chunk and sort best score first."""
    qa = get_qa()
    ranked = []
    for d in docs:
        r = qa(question=question, context=d.page_content)
        ranked.append(
            {
                "answer": r["answer"],
                "score": float(r.get("score", 0.0)),
                "context": d.page_content,
                "page": int(d.metadata.get("page", 0)) + 1,
                "source": d.metadata.get("source", ""),
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def log_feedback(question, item, helpful):
    new_file = not FEEDBACK_LOG.exists()
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "question", "answer", "score", "page", "source", "helpful"])
        w.writerow(
            [
                datetime.datetime.now().isoformat(timespec="seconds"),
                question,
                item["answer"],
                round(item["score"], 4),
                item["page"],
                item["source"],
                helpful,
            ]
        )


def show(item, question):
    answer = item["answer"]
    if item["score"] < LOW_CONFIDENCE:
        answer = f"I'm not very sure, but my best guess is: {answer}"
    return render_template(
        "index1.html",
        answer=answer,
        context=item["context"],
        question=question,
        source_page=item["page"],
        score=round(item["score"], 3),
        source_file=item["source"],
    )


def msg(text):
    return render_template("index1.html", answer=text)


# ---------------------------------------------------------------- routes
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index2.html")


@app.route("/upload", methods=["POST"])
def upload():
    question = (request.form.get("textInput") or "").strip()
    files = [f for f in request.files.getlist("pdfFile") if f and f.filename]

    if not question:
        return msg("Please enter a question.")
    if not files:
        return msg("Please choose a PDF file.")
    if not all(allowed_file(f.filename) for f in files):
        return msg("Please upload PDF files only.")

    try:
        pages = load_pdfs(files)
        if not pages:
            return msg("The PDF contains no readable text.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        db = FAISS.from_documents(docs, get_embeddings())
        matches = db.similarity_search(question, k=TOP_K)
        if not matches:
            return msg("I couldn't find relevant information in that PDF.")

        ranked = rank_passages(question, matches)
        app.logger.info("Best answer score: %.3f", ranked[0]["score"])

        sid = uuid.uuid4().hex
        STORE[sid] = {"question": question, "items": ranked, "index": 0}
        session["sid"] = sid

        return show(ranked[0], question)

    except Exception:
        app.logger.exception("Failed to process uploaded PDF")
        return msg("I couldn't read or process that PDF. Please try another file.")


@app.route("/upload/feedback", methods=["POST"])
def feedback():
    state = STORE.get(session.get("sid"))
    if not state:
        return msg("Please upload a PDF and ask a question first.")

    question = state["question"]
    items = state["items"]
    current = items[state["index"]]

    # Positive feedback ("satisfied")
    if request.form.get("btnTask2") == "click2":
        log_feedback(question, current, helpful=True)
        STORE.pop(session.get("sid"), None)
        session.pop("sid", None)
        return msg("Thank you for your positive feedback.")

    # Negative feedback -> show next best passage
    if request.form.get("btnTask3") == "click3":
        log_feedback(question, current, helpful=False)
        state["index"] += 1

        if state["index"] >= len(items):
            STORE.pop(session.get("sid"), None)
            session.pop("sid", None)
            return msg("No more relevant passages were found. Please try another question.")

        return show(items[state["index"]], question)

    return msg("Please choose a feedback option.")


if __name__ == "__main__":
    # debug=True is for development only. Turn it off when deploying.
    app.run(debug=True)
