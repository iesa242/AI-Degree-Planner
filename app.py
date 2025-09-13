import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
from dotenv import load_dotenv

# Load AWS credentials
load_dotenv()

EMBED_MODEL = "amazon.titan-embed-text-v1"
LLM_MODEL = "us.amazon.nova-micro-v1:0"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

st.set_page_config(page_title="AI Degree Planner", page_icon="🎓")
st.title("🎓 AI Degree Planner")

# Upload transcript + catalog
uploaded_files = st.file_uploader("Upload Transcript + Major Catalog (PDF/TXT)", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file)
        else:
            loader = TextLoader(file)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL, region_name=AWS_REGION)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_schedule")

    st.success("Transcript + Catalog uploaded and indexed ✅")

    # User input: units per semester
    units = st.number_input("How many units per semester?", min_value=6, max_value=21, step=1)

    if st.button("Generate Graduation Plan"):
        llm = Bedrock(model_id=LLM_MODEL, region_name=AWS_REGION)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents("degree requirements")])

        prompt = f"""
        You are an academic advisor AI. Based on this transcript and major catalog:
        {context}

        The student wants to take {units} units per semester.
        Create a semester-by-semester class plan until graduation.
        Show output as a table with columns: Semester, Classes, Total Units.
        Estimate graduation semester.
        """

        response = llm.invoke(prompt)
        st.write("## 📅 Suggested Graduation Plan")
        st.write(response)
