from typing import List

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore


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


def main():
    load_dotenv()

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:"
    )

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents:")

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

                # store to vector database
                vectorstore: VectorStore = get_vectorstore(text_chunks)


if __name__ == '__main__':
    main()
