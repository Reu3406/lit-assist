import streamlit as st
from PyPDF2 import PdfReader
import pypdf
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# for summarising
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import re
import pandas as pd
import os

import path
import sys


file_name = []
answer = []

file_match_chunk_df = pd.DataFrame({"file": [], "page no.": [], "excerpt": []})


def main():

    st.set_page_config(page_title="AI document intepreter")

    # if "openai_key" not in st.session_state:
    #     st.session_state.openai_key = None

    st.header("Upload Your PDFs and ask me anything about them")
    user_question = st.text_input("What would you like to know from your documents?")
    if user_question:
        query = user_question

    with st.sidebar:
        st.subheader("Your OpenAI API key:")
        KEY = st.text_input("OpenAI API Key")
        # if KEY:
        #     st.session_state.openai_key = KEY
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            type="pdf",
            accept_multiple_files=True,
        )
        for pdf in pdf_docs:
            # locating local machine filepath address
            dir = path.Path(__file__).abspath()
            sys.path.append(dir.parent.parent)
            # reference using  ./
            with open(os.path.join("./pdf_docs", uploadedfile.name), "wb") as f:
                f.write(uploadedfile.getbuffer())

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0,
        openai_api_key=KEY,
    )

    chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=KEY)

    summary_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=KEY,
    )

    def pdf_to_vectorstore(pdf):
        pdf_reader = PyPDFLoader(pdf)

        pages = pdf_reader.load_and_split(text_splitter=text_splitter)

        vectorstore = FAISS.from_documents(documents=pages, embedding=embeddings)
        file_name.append(re.sub(r".*/", "", pdf_reader.file_path))
        return vectorstore

    def query_vectorstore_match(query, vectorstore, k=3):
        matches = vectorstore.similarity_search(query, k=k)

        file = []
        page = []
        content = []
        for match in matches:
            file.append(re.sub(r".*/", "", match.metadata["source"]))
            page.append(match.metadata["page"] + 1)
            content.append(match.page_content)

        dict = {"file": file, "page no.": page, "excerpt": content}
        result = pd.DataFrame(dict)

        file_match_chunk_df = pd.concat(
            [file_match_chunk_df, result], ignore_index=True
        )
        return matches

    def answer_query(matches, query):
        answer.append(chain.run(input_documents=matches, question=query))
        return

    def summarise_answers(answer):
        text_to_summarize = "\n".join(answer)
        # Make the API call to OpenAI GPT-3 for summarization
        chat_messages = [
            SystemMessage(
                content=f"You are an academic research expert tasked to answer the following question: '{query}'"
            ),
            HumanMessage(
                content=f"answer the question by providing a single concise summary of the following text: '{text_to_summarize}'. \n Keep the summary within 1000 words "
            ),
        ]
        summary = summary_llm(chat_messages).content
        return summary

    if st.button("Process"):
        with st.spinner("Processing"):
            for pdf in [dir + f for f in os.listdir("./pdf_docs")]:
                vectorstore = pdf_to_vectorstore(pdf)
                matches = query_vectorstore_match(query, vectorstore, k=3)
                answer_query(matches, query)

            file_answer_df = pd.DataFrame({"file": file_name, "answer": answer})
            summary = summarise_answers(answer)
            st.write(summary)
            st.dataframe(file_answer_df, hide_index=True)
            st.dataframe(file_match_chunk_df, hide_index=True)


if __name__ == "__main__":
    main()
