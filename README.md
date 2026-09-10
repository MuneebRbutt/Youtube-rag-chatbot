# YouTube Transcript RAG

Ask questions about the English transcript of a YouTube video, or generate a
full-video summary. Questions use OpenAI embeddings and FAISS similarity search;
summaries use every transcript chunk in a hierarchical map-reduce workflow.

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

## Streamlit Frontend

Launch the non-technical browser interface:

```powershell
streamlit run streamlit_app.py
```

Paste one or more YouTube URLs, select similarity or MMR retrieval, then ask
questions or use **Summarize All Processed Videos**. The frontend reads the
OpenAI key from the local `.env` file and never displays it.

To protect the API budget, each submitted video may be up to 20 minutes long
and all videos in one submission may total up to 30 minutes. Duration is
estimated from the final English-caption timestamp, and over-limit submissions
are rejected before any OpenAI embedding or summarization request is made.

### Deploy to Streamlit Community Cloud

Do not commit `.env` or any API key to GitHub. A deployed Streamlit app does
not receive your local `.env` file. After deployment, open **Settings** →
**Secrets** for the app and add:

```toml
OPENAI_API_KEY = "sk-your-real-key"
```

Save the secret, then reboot or redeploy the app. Streamlit Community Cloud
exposes root-level secrets as environment variables, so the application can
read this key securely. See the [Streamlit secrets documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

Ask a custom question:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --question "What are the main points?"
```

Build a shared knowledge base from multiple videos and ask a cross-video
question:

```powershell
python app.py "https://youtu.be/VIDEO_ID_1" "https://youtu.be/VIDEO_ID_2" --index-dir .\data\knowledge-base --question "How do these videos explain fine-tuning differently?"
```

Add more videos to an existing knowledge base:

```powershell
python app.py "https://youtu.be/VIDEO_ID_3" --index-dir .\data\knowledge-base --add-to-index
```

Generate a full-video summary. This does not use FAISS top-k retrieval; every
transcript chunk is summarized before the final summary is produced:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --mode summary
```

Optionally persist a FAISS index locally. Reusing the same `--index-dir` loads
that index instead of downloading and embedding the transcript again:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --index-dir .\data\video-index
```

## Compare Retrieval Modes

Question mode uses similarity search by default. Build an index once, then run
the same question with each retrieval mode to compare relevance and diversity:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --index-dir .\data\video-index --question "What are the main points?"
python app.py "https://youtu.be/VIDEO_ID" --index-dir .\data\video-index --retrieval-mode mmr --retrieval-k 4 --mmr-fetch-k 12 --question "What are the main points?"
```

MMR considers `--mmr-fetch-k` candidate chunks and returns the most diverse
`--retrieval-k` chunks. It is available only for question answering; full-video
summary mode always processes every transcript chunk.

## Follow-Up Questions

Start an interactive question-answering session with `--chat`:

```powershell
python app.py "https://youtu.be/VIDEO_ID" --chat
```

The application rewrites follow-up questions into standalone retrieval queries.
For example, after asking about backpropagation, `Why is it important?` is
rewritten before retrieval. Recent conversation is used only for rewriting; it
is not inserted into the vector search query.

Errors shown in the terminal are user-safe. Detailed diagnostics are recorded
in `rag_app.log`.

Each answer includes clickable source citations. Each citation opens the
retrieved point in the original YouTube video, for example:

```text
Sources:
- [Video title at 18:42](https://www.youtube.com/watch?v=VIDEO_ID&t=1122s)
```

## Limitations

- The app works only with public YouTube videos that have an accessible English transcript. Private, deleted, unavailable, caption-disabled, or non-English-only videos cannot be processed.
- Video titles are retrieved through YouTube's public oEmbed endpoint. If YouTube does not return a title, processing stops even if a transcript might otherwise be available.
- The Streamlit knowledge base exists only in the current browser session. Refreshing, restarting, or processing a new URL set replaces the prior videos and clears chat history. Use the CLI with `--index-dir` and `--add-to-index` for local persistent multi-video indexes.
- Full-video summaries process every transcript chunk with map-reduce. They can be slow and consume substantially more OpenAI tokens than retrieval-based Q&A.
- Retrieval quality depends on transcript quality, chunking, embedding quality, and the selected similarity or MMR settings. Answers can only be as complete as the retrieved transcript context.
- The app has no user authentication, rate limiting, usage quotas, database, or shared persistent knowledge base. Do not expose a public deployment without adding appropriate access and cost controls.
- FAISS indexes are stored locally by the CLI and should be loaded only from trusted locations because FAISS persistence uses deserialization.

## Tests

```powershell
python -m unittest discover -s tests
```
