
# Local Chatbot Demo (RAG-style)

This is a simple FastAPI-based chatbot demo using placeholder logic. You can extend it to include LangChain, vector stores, and ElevenLabs TTS.

## 🧪 How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Start the server:
```
python main.py
```

3. Test the chatbot (use Postman or curl):
```
POST http://localhost:8000/ask
Body: { "question": "What is Rajkumar's experience?" }
```

You will get a JSON response with a text and voice placeholder.

## 📁 Included

- `main.py`: FastAPI backend
- `dataset.txt`: Extracted content
- `requirements.txt`: Dependencies
