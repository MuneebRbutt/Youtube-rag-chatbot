# YouTube Transcript RAG

Ask questions about the English transcript of a YouTube video. The application
uses OpenAI embeddings, FAISS similarity search, and `gpt-4o-mini` to answer
only from retrieved transcript context.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your real OpenAI key in `.env`:

```ini
OPENAI_API_KEY=your_actual_key
```

## Run

```powershell
python app.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Ask a custom question:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --question "What are the main points?"
```

Optionally persist a FAISS index locally. Reusing the same `--index-dir` loads
that index instead of downloading and embedding the transcript again:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --index-dir .\data\video-index
```

Errors shown in the terminal are user-safe. Detailed diagnostics are recorded
in `rag_app.log`.

## Tests

```powershell
python -m unittest discover -s tests
```
