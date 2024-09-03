import os
import warnings
warnings.filterwarnings("ignore")

# pip install langchain langchain-google-genai langchain-community langchain-chroma langchain-huggingface python-dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# Document to retrieve data from
document = "Resume.txt"

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
file_path = os.path.join(current_dir, document)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create database if not exists
if not os.path.exists(persistent_directory):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = Chroma.from_documents(docs, embeddings_model, persist_directory=persistent_directory)

else :
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings_model)

# Set up retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Define llm
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ---------------------------------------------------------------

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever that uses llm to create context aware prompt that is send to-
# the retreiver to retreive relevant data and return context
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. If you don't know the answer, use "
    "the following pieces of retrieved context to answer the"
    "question, else just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# create_retrieval_chain sends the context from history_aware_retriever to the create_stuff_documents_chain that-
# combines input and context and sends it to llm

# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit()' to end the conversation.")
    chat_history = []  
    while True:
        query = input("You: ")
        if query.lower() == "exit()":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        print(f"AI: {result['answer']}")

        chat_history.append(("human", query))
        chat_history.append(("ai", result["answer"]))


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
