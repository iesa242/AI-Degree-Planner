import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from dotenv import load_dotenv

# Load AWS credentials
load_dotenv()

# AWS Bedrock configs
EMBED_MODEL = "amazon.titan-embed-text-v1"
LLM_MODEL = "us.amazon.nova-micro-v1:0"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

# Streamlit config
st.set_page_config(page_title="AI Degree Planner", page_icon="🎓")
st.title("🎓 AI Degree Planner")

st.markdown("Upload your **Transcript** and **Major Catalog** to generate a semester-by-semester plan.")

# Layout with two upload columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Transcript")
    transcript_file = st.file_uploader("Upload student transcript (PDF)", type=["pdf"], key="transcript")

with col2:
    st.subheader("📚 Upload Major Catalog")
    catalog_file = st.file_uploader("Upload major catalog (PDF/TXT)", type=["pdf", "txt"], key="catalog")

# Placeholders for docs
transcript_docs, catalog_docs = [], []

# Process transcript
if transcript_file:
    loader = PyPDFLoader(transcript_file)
    transcript_docs = loader.load()

# Process catalog
if catalog_file:
    if catalog_file.name.endswith(".pdf"):
        loader = PyPDFLoader(catalog_file)
    else:
        loader = TextLoader(catalog_file)
    catalog_docs = loader.load()

# Combine docs for knowledgebase
all_docs = transcript_docs + catalog_docs

if all_docs:
    st.success("✅ Documents uploaded successfully")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Build embeddings + ChromaDB
    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL, region_name=AWS_REGION)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_degree_planner")

    # Units per semester input
    units = st.number_input("How many units per semester?", min_value=6, max_value=21, step=1)

    if st.button("Generate Graduation Plan"):
        llm = Bedrock(model_id=LLM_MODEL, region_name=AWS_REGION)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Get context from both transcript + catalog
        transcript_text = "\n".join([doc.page_content for doc in transcript_docs])
        catalog_text = "\n".join([doc.page_content for doc in catalog_docs])

        prompt = f"""
        You are an academic advisor AI.

        Student transcript:
        {transcript_text}

        Major catalog (degree requirements):
        {catalog_text}

        The student wants to take {units} units per semester.

        1. Identify which classes are already completed (from transcript).
        2. Identify which classes are still required (from catalog).
        3. Create a semester-by-semester class plan until graduation.
        4. Show output as a table with columns: Semester, Classes, Total Units.
        5. Estimate the graduation semester.

        Answer clearly and concisely.
        """

        response = llm.invoke(prompt)

        st.write("## 📅 Suggested Graduation Plan")
        st.write(response)
else:
    st.info("⬆️ Please upload both **Transcript** and **Major Catalog** to continue.")
