from typing import List

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from static.htmlTemplates import css, bot_template, user_template


def get_pdfs_text(pdfs) -> str:
    return ''.join([get_pdf_text(pdf) for pdf in pdfs])


def get_pdf_text(pdf) -> str:
    pdf_reader = PdfReader(pdf)
    return ''.join([page.extract_text() for page in pdf_reader.pages])


def get_text_chunks(raw_text) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(raw_text)


def get_vectorstore(text_chunks: List[str]) -> VectorStore:
    embeddings: Embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore: VectorStore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_user_input(user_question: str):
    response = st.session_state.conversation(
        {'question': user_question}
    )


def main():
    load_dotenv()

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello bot"),
             unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"),
             unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf texts
                raw_text = get_pdfs_text(pdfs)

                # chunk texts
                text_chunks: List[str] = get_text_chunks(raw_text)

                # create vector store
                vectorstore: VectorStore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
