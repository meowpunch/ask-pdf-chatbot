import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def get_pdfs_text(pdfs):
    return ''.join([get_pdf_text(pdf) for pdf in pdfs])


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    return ''.join([page.extract_text() for page in pdf_reader.pages])


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(raw_text)


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
                text_chunks = get_text_chunks(raw_text)

                # store to vector database


if __name__ == '__main__':
    main()
