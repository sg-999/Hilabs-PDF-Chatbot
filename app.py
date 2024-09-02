import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import huggingface_pipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS
from accelerate import load_checkpoint_and_dispatch
from chromadb.config import Settings

checkpoint = ".\LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
persist_directory = "db"
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory ="db",
    anonymized_telemetry = False
)

# Specify the offload folder
# offload_folder = os.path.join(".", "offload_folder")

# # Ensure the offload folder exists
# os.makedirs(offload_folder, exist_ok=True)
offload_folder = os.path.abspath(".\\offload_folder")

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model directly with its weights
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Load the checkpoint and dispatch to the appropriate device
base_model = load_checkpoint_and_dispatch(
    base_model,
    checkpoint=checkpoint,
    device_map="auto",
    offload_folder = os.path.abspath(".\\offload_folder")  # Ensure this path is correct
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    # local_llm = huggingface_pipeline(pipeline=pipe)
    return pipe



@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
     #Define your documents here
    # documents = [
    #     {"text": "This is the first document content.", "metadata": {"source": "doc1.txt"}},
    #     {"text": "This is the second document content.", "metadata": {"source": "doc2.txt"}}
    # ]
    documents = [document["content"] for document in documents]

    output_dir = "./db_metadata_v5"
    db = Chroma.from_documents(documents=documents, embedding_function=embeddings, persist_directory=output_dir)
    # Create Chroma instance with documents

    # db = Chroma.from_documents(documents=docs,embedding_function=embeddings, persist_directory=output_dir)
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_document=True
    )
    return qa

def process_answer(instruction):
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

documents = []
for root, dirs, files in os.walk("docs"):
    for file in files:
        if file.endswith(".pdf"):
            print(f"Loading file: {file}")
            loader = PDFMinerLoader(os.path.join(root, file))
            documents.extend(loader.load())

if not documents:
    print("No documents found.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

if not texts:
    print("No texts found after splitting documents.")

# Create embeddings here
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
try:
    db = Chroma.from_documents(
        texts, embeddings, persist_directory=persist_directory
    )
except IndexError as e:
    print(f"Error while creating embeddings: {e}")
    print(f"Texts: {texts}")

st.title('Search your PDF')
with st.expander('About the App'):
    st.markdown(
        """
        This is a Generative AI powered Question and Answering App
        """
    )
    question = st.text_area("Enter your Question")
    if st.button("Search"):
        st.info("Your Question: " + question)
        st.info("Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

