from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import tempfile

import os

from qdrant_client import QdrantClient

qdrant_url = os.getenv('QDRANT_URL', default='localhost')
qdrant_port = os.getenv('QDRANT_PORT', default=6333)
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = QdrantClient(host=qdrant_url, port=qdrant_port)


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf.read())
            st.write(f"File saved to {tmp_file.name}")

        loader = PyPDFLoader(tmp_file.name)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        pages = loader.load_and_split(text_splitter)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        qdrant = Qdrant.from_documents(pages, embeddings, url=qdrant_url, collection_name=COLLECTION_NAME)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = qdrant.similarity_search(query=user_question, k=4)
            st.write(docs)
            llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
