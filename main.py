import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()
from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# -------------------------------
# File paths
# -------------------------------
document = "Resume.txt"

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "faiss_db")
file_path = os.path.join(current_dir, document)

# -------------------------------
# Embeddings
# -------------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# -------------------------------
# Create / Load Vector DB
# -------------------------------
if not os.path.exists(persistent_directory):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # FIXED chunk size (IMPORTANT)
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local(persistent_directory)
else:
    db = FAISS.load_local(
        folder_path=persistent_directory,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True
    )

# -------------------------------
# Retriever
# -------------------------------
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# -------------------------------
# LLM (Ollama)
# -------------------------------
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0,
    top_p=0.1
)

# -------------------------------
# Contextualize Question Prompt
# -------------------------------
contextualize_q_system_prompt = (
    "Given chat history and latest user question, "
    "rewrite the question so it is standalone. "
    "Do NOT answer it."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------------------
# History-aware retriever
# -------------------------------
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# -------------------------------
# QA Prompt
# -------------------------------
qa_system_prompt = (
    "You are a QA assistant. Use retrieved context to answer. "
    "If unknown, say you don't know. Max 3 sentences.\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------------------
# QA Chain
# -------------------------------
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# -------------------------------
# LCEL RAG PIPELINE (MODERN)
# -------------------------------
rag_chain = (
    {
        "context": history_aware_retriever,
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history"),
    }
    | question_answer_chain
)

# -------------------------------
# Chat loop
# -------------------------------
def continual_chat():
    print("Start chatting with the AI! Type 'exit()' to end.")
    chat_history = []

    while True:
        query = input("You: ")

        if query.lower() == "exit()":
            break

        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })

        print(f"AI: {result}")

        # UPDATED message format
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result))

# -------------------------------
# Debug retriever
# -------------------------------
def retrieve_docs():
    while True:
        prompt = input("You: ")
        if prompt == "exit()":
            break

        docs = retriever.invoke(prompt)

        print(f"\nRetrieved {len(docs)} docs:\n")
        for doc in docs:
            print(doc.page_content)
            print("-" * 50)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    continual_chat()