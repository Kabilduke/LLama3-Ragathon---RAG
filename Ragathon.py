import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama

local_llm = "llama3"
llm = ChatOllama(model= local_llm, temperature=0)

st.title("Pdf Data Analyser")

uploaded_file = st.file_uploader("PDF files only", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"Number of pages: {num_pages}")

    text_content = ""

    for page_no in range(num_pages):
        page_text = pdf_reader.pages[page_no].extract_text()

        text_content += page_text

        #st.write(f"Page {page_no + 1} Content:")
        #st.write(text_content)

    # Text splitting into chunk text
    splitter  = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 250, chunk_overlap = 0)

    text_split = splitter.split_text(text_content)

    #Adding vector to text
    vectorstore = Chroma.from_texts(
        texts = text_split,
        collection_name= 'rag-chroma',
        embedding= GPT4AllEmbeddings()
        )
    retriever = vectorstore.as_retriever()

    #User query
    user_question = st.text_input("Ask a question:")

    if st.button("Get Answers"):
        response = llm.generate(question = user_question, content = text_content)
        st.write("RAG response: ")
        st.write(response)







