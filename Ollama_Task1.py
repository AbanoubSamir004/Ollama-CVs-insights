import streamlit as st 
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from PIL import Image
import os
import PyPDF2


def OllamaModel():
    DATA_PATH = "CVs"  

    def load_documents():
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()

    data = load_documents()
    
    # Split and chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunks = text_splitter.split_documents(data)

    # Add to vector database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="llama3", show_progress=True),
        collection_name="local-rag"
    )

    # LLM from Ollama
    local_model = "llama3"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to answer user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, vector_db

def get_or_init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return st.session_state["chat_history"]

def append_to_chat_history(user_question, chat_response):
    chat_history = get_or_init_chat_history()
    chat_history.append({"question": user_question, "response": chat_response})
    st.session_state["chat_history"] = chat_history

def display_chat_history():
    chat_history = get_or_init_chat_history()
    for chat in chat_history:
        st.text_area("You:", value=chat["question"], height=100, max_chars=None, key=None)
        st.text_area("ChatGPT:", value=chat["response"], height=200, max_chars=None, key=None)

# Streamlit UI
def main(chain):
    # Set page background color
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #333333;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("")
    st.title("ChatGPT with Ollama Demo")
    st.markdown("Welcome to ChatGPT with Ollama! Feel free to ask me anything.")
    
    display_chat_history()

    # Input box for user questions
    user_question = st.text_input("You:", key="input")

    if st.button("Ask") or st.session_state.get("ask_pressed", False):
        st.session_state["ask_pressed"] = False
        # Add a waiting spinner while processing
        with st.spinner("Processing..."):

            # Invoke OllamaModel
            response = chain.invoke(user_question)
            append_to_chat_history(user_question, response)
            st.text_area("ChatGPT:", value=response, height=200, max_chars=None, key=None)

    else:
        st.session_state["ask_pressed"] = True


if __name__ == "__main__":

    chain,vector_db = OllamaModel()
    main(chain)

