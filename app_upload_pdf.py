import streamlit as st
import os
import fitz  # PyMuPDF for PDF rendering
import base64
from datetime import datetime
from util import LlamaHelper
from diskcache import Cache
from pathlib import Path

cache = Cache('./cache')
my_helper = LlamaHelper(cache, logger='Info', chunk_size=100)
service_context = my_helper.service_context

def highlight_text_in_pdf(pdf_path, source_texts, selected_pdf):
    pdf_document = fitz.open(pdf_path)
    for page in pdf_document:
        annot = page.first_annot
        while annot:
            annot = page.delete_annot(annot)
    for source_text in source_texts:
        start = source_text[:3]
        end = source_text[-3:]
        for page in pdf_document:
            rl1 = page.search_for(start)
            # print(rl1)
            if rl1 == []:  # not on page
                continue
            rl2 = page.search_for(end)
            if rl2 == []:  # not on page
                continue
            pointa = rl1[0].tl
            pointb = rl2[0].br
            print(pointa, pointb)
            page.add_highlight_annot(start=pointa, stop=pointb)

    pdf_document.save(f"./temp/temp_{selected_pdf}")
    pdf_document.close()

def save_uploaded_file(uploaded_file):
    # Create a directory for storing uploaded HTML files
    os.makedirs("uploaded_files", exist_ok=True)

    # Save the uploaded HTML file
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

# Streamlit App Header
st.title("Ask your PDF")

# File upload for PDF
st.header("Upload a PDF File")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Main content
if uploaded_file:
    st.write("Uploaded PDF:")
    st.write(uploaded_file.name)
    uploaded_file_name = uploaded_file.name
    uploaded_file_loc = save_uploaded_file(uploaded_file)
    uploaded_file_loc = uploaded_file_loc.replace("\\", "/")
    # pdf_path = f'./pdf/{selected_folder}/{selected_pdf}'
    index_files = os.listdir('./index')
    print(uploaded_file_name)
    print(uploaded_file_loc)
    if f'{uploaded_file_name.split(".")[0]}' not in index_files:
        with st.spinner("Building index for PDF..."):
            my_helper.make_index_pdf(uploaded_file_loc)

    index = my_helper.load_index(f'./index/{uploaded_file_name.split(".")[0]}')
    query_engine = my_helper.get_query_engine_for_index(index)

    st.header("Questions and Answers")
    question = st.text_area("Ask a question:")
    if st.button("Submit Question"):
        st.write("Your Question:", question)
        response = query_engine.query(question)
        st.write("Your Answer:", response)
        source_texts = []
        for node in response.source_nodes:
            source_text = node.node.text
            source_texts.append(source_text)

        highlight_text_in_pdf(uploaded_file_loc, source_texts, uploaded_file_name)

        # st.write("PDF Preview with Highlights:")
        # with open(temp_pdf_path, "rb") as pdf_file:
        #     st.write(pdf_file.read())

    # Display PDF using PyMuPDF
    pdf_path = Path(uploaded_file_loc)
    # print(os.listdir('./temp'))
    # print(selected_pdf)
    if f'temp_{uploaded_file_name}' in os.listdir('./temp'):
        pdf_path = Path(f'./temp/temp_{uploaded_file_name}')
    print(pdf_path)
    st.write("PDF Preview:")
    base64_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="800px" height="2100px" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Save Q&A to a text file
    if st.button("Save Q&A"):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        txt_file_name = f"Q&A_{current_time}.txt"
        with open(txt_file_name, "w") as txt_file:
            txt_file.write(f"Question: {question}\n")
            txt_file.write(f"Answer: {response.text}\n")
        st.success(f"Q&A saved to {txt_file_name}")
