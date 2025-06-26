# Local Chatbot Demo (RAG-style)

This is a simple FastAPI-based chatbot demo using placeholder logic. You can extend it to include LangChain, vector stores, and ElevenLabs TTS.

## 🧪 How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python main.py
   ```

3. **Test the chatbot (use Postman or curl):**
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is Rajkumar\'s experience?"}'
   ```
   You will get a JSON response with a text and voice placeholder.

## 🚀 Deploying on Render

1. **Pin Python Version:**
   - Create a file named `runtime.txt` in your project root with:
     ```
     python-3.11.9
     ```
   - This ensures compatibility with all dependencies.

2. **Deploy:**
   - Connect your repo to Render and deploy as a web service.
   - Set the start command to:
     ```
     python main.py
     ```

3. **Troubleshooting:**
   - If you see errors about `numpy` or other dependencies, make sure you are not using Python 3.13+.
   - Only use stable versions in `requirements.txt` (no `rc` or pre-releases).
   - If you see build errors, check the Render logs and ensure your Python version is pinned as above.

## 📁 Included

- `main.py`: FastAPI backend
- `dataset.txt`: Extracted content
- `requirements.txt`: Dependencies
- `runtime.txt`: (Recommended) Pin Python version for deployment

## 🛠️ Extending

- Integrate LangChain for advanced RAG workflows
- Add a vector store for semantic search
- Integrate ElevenLabs or other TTS APIs for voice output

---

**If you encounter deployment issues, check your Python version and dependency versions first.**
