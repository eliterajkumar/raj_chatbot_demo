from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

def clean_output(response: str) -> str:
    """
    Cleans the LLM's raw output.
    Ensures any residual thought tags are removed and formats whitespace.
    """
    cleaned = response.strip()
    cleaned = ' '.join(cleaned.splitlines()).strip()
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned[len("answer:"):].strip()
    return cleaned

def load_documents(file_path="fynorra_dataset.txt"):
    """Loads and splits documents from a text file."""
    loader = TextLoader(file_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(raw_docs)

def setup_vectorstore(docs, persist_dir="db"):
    """Sets up and persists the Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    return vectordb

def setup_llm():
    """
    Initializes the Ollama Large Language Model with adjusted parameters
    for better direct answer generation.
    """
    return OllamaLLM(
        model="llama3:latest",
        temperature=0.3,
        num_predict=500, # Increased max tokens for more complete answers
        top_p=0.9,
        # Refined stop sequences to prevent internal monologues and cutoffs
        stop=["\nQuestion:", "\nUser:", "<think>", "</think>", "\nAnswer:"]
    )

# CONDENSE_QUESTION_PROMPT is still needed by ConversationalRetrievalChain,
# even if chat_history is empty. It will effectively rephrase the current question
# into itself.
CONDENSE_QUESTION_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You should not answer the question, just rephrase it.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

ANSWER_GENERATION_PROMPT = """You are a helpful and knowledgeable AI assistant for Fynorra, an AI and IT solutions company.
Your goal is to provide a concise, accurate, and helpful answer based *only* on the provided context.
**Immediately provide the answer directly.** Do not include any preambles or internal thoughts.
Keep your answer professional and to the point, ideally under 50 words.
If the information is not directly available in the provided context, state that you don't have enough information and suggest they ask about core services.
Do not make up information.

Context:
{context}

Question: {question}
Answer:"""


def build_qa_chain():
    """Builds and returns the conversational QA chain (now with no history assumed)."""
    docs = load_documents()
    vectordb = setup_vectorstore(docs)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    llm = setup_llm()

    condense_question_prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_QUESTION_PROMPT,
    )

    answer_generation_prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=ANSWER_GENERATION_PROMPT,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_question_prompt_template,
        combine_docs_chain_kwargs={"prompt": answer_generation_prompt_template},
        return_source_documents=False
    )

    # The chat_history parameter remains in the function signature,
    # but main.py will now always pass an empty list.
    def ask_cleaned(question: str, chat_history: list = []) -> str:
        """Runs the QA chain and cleans the output."""
        raw_output = chain.invoke({"question": question, "chat_history": chat_history})
        cleaned = clean_output(raw_output['answer'])

        words = cleaned.split()
        if len(words) > 50:
            cleaned = " ".join(words[:50]) + "..."

        return cleaned

    return ask_cleaned

qa_chain = build_qa_chain()

if __name__ == "__main__":
    print("Fynorra Chatbot Ready! Type 'exit' to quit.")
    chat_history = [] # This will always remain empty now for CLI test
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = qa_chain(user_input, chat_history) # Pass empty history
        print(f"Fynorra Bot: {response}")
        
        # We no longer append to chat_history for persistence in CLI either
        # chat_history.append((user_input, response))
