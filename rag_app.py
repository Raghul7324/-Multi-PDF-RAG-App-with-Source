import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.title("Chat with your PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

pdf_files = [(file.name, file.getvalue()) for file in uploaded_files]
st.session_state["pdf_files"] = pdf_files

file_signature = tuple(
    sorted(
        (name, hashlib.sha256(contents).hexdigest())
        for name, contents in pdf_files
    )
)

if st.button("Build or rebuild FAISS index"):
    if not pdf_files:
        st.warning("Upload at least one PDF first.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("Set the OPENAI_API_KEY environment variable before building the index.")
    else:
        documents = []

        with st.spinner("Reading uploaded PDFs..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for index, (file_name, contents) in enumerate(pdf_files):
                    pdf_path = Path(temp_dir) / f"{index}_{Path(file_name).name}"
                    pdf_path.write_bytes(contents)

                    for document in PyPDFLoader(str(pdf_path)).load():
                        document.metadata["source"] = file_name
                        document.metadata["page"] = int(
                            document.metadata.get("page", 0)
                        ) + 1
                        documents.append(document)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        if not chunks:
            st.error("No readable text was found in the uploaded PDFs.")
        else:
            with st.spinner("Embedding PDF chunks and building one FAISS index..."):
                embeddings = OpenAIEmbeddings(
                    api_key=os.environ["OPENAI_API_KEY"]
                )
                st.session_state["vector_store"] = FAISS.from_documents(
                    chunks,
                    embeddings,
                )
                st.session_state["indexed_file_signature"] = file_signature
                st.session_state["indexed_chunk_count"] = len(chunks)

            st.success(
                f"Finished building one FAISS index with {len(chunks)} chunks "
                f"from {len(pdf_files)} PDF(s)."
            )

if "vector_store" in st.session_state:
    if st.session_state.get("indexed_file_signature") == file_signature:
        st.caption(
            f"Current index: {st.session_state['indexed_chunk_count']} chunks."
        )
    else:
        st.warning("The uploaded PDFs changed. Rebuild the FAISS index before asking.")

question = st.text_input("Ask a question about the uploaded PDFs")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question first.")
    elif (
        "vector_store" not in st.session_state
        or st.session_state.get("indexed_file_signature") != file_signature
    ):
        st.warning("Upload PDFs and build the FAISS index first.")
    else:
        retrieved_chunks = st.session_state["vector_store"].similarity_search(
            question,
            k=4,
        )

        context = "\n\n".join(
            f"Source: {chunk.metadata.get('source', 'Unknown')}, "
            f"page: {chunk.metadata.get('page', 'Unknown')}\n"
            f"{chunk.page_content}"
            for chunk in retrieved_chunks
        )

        prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the supplied PDF passages. "
            "If they do not contain the answer, say so. Cite supporting "
            "passages using their source file name and page number.\n\n"
            "PDF passages:\n{context}\n\n"
            "Question: {question}"
        )
        answer_chain = (
            prompt
            | ChatOpenAI(
                model="gpt-4o-mini",
                api_key=os.environ["OPENAI_API_KEY"],
            )
            | StrOutputParser()
        )
        answer = answer_chain.invoke(
            {"context": context, "question": question}
        )

        st.subheader("Answer")
        st.write(answer)

        sources = list(
            dict.fromkeys(
                (
                    chunk.metadata.get("source", "Unknown"),
                    chunk.metadata.get("page", "Unknown"),
                )
                for chunk in retrieved_chunks
            )
        )
        st.markdown("**Retrieved sources**")
        for file_name, page in sources:
            st.markdown(f"- `{file_name}`, page {page}")
