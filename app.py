import streamlit as st
from PyPDF2 import PdfReader

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

from langchain.docstore.document import Document

if "openai_key" not in st.session_state:
    st.session_state.openai_key = "Invalid"


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    openai_api_key=st.session_state.openai_key,
)

chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=st.session_state.openai_key,
)

summary_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=st.session_state.openai_key,
)


file_name = []

answer = []

file_match_chunk_df = pd.DataFrame({"file": [], "page no.": [], "excerpt": []})


def main():

    file_name = []

    answer = []

    file_match_chunk_df = pd.DataFrame({"file": [], "page no.": [], "excerpt": []})

    st.set_page_config(page_title="AI multi-document RAG summarise")

    st.header(
        "Enter your OpenAI API key - Upload Your PDFs - and ask me anything about them"
    )
    user_question = st.text_input("What would you like to know from your documents?")

    with st.sidebar:
        st.subheader("Your OpenAI API key:")
        KEY = st.text_input("please enter you OpenAI API key")
        if KEY:
            st.session_state.openai_key = KEY

            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and proceed to input your question and click 'Process'",
                type="pdf",
                accept_multiple_files=True,
            )

    if user_question and st.button("Process"):
        with st.spinner("Processing"):
            query = user_question
            docs = []
            file_name = []

            for n, pdf in enumerate(pdf_docs):
                file_name.append(pdf.name)
                reader = PdfReader(pdf)

                for i, page in enumerate(reader.pages):
                    docs.append(
                        Document(
                            page_content=page.extract_text(),
                            metadata={"page": i + 1, "source": file_name[n]},
                        )
                    )
                vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

                matches = vectorstore.similarity_search(query, k=3)

                file = []
                page = []
                content = []

                for match in matches:
                    file.append(re.sub(r".*/", "", match.metadata["source"]))
                    page.append(match.metadata["page"])
                    content.append(match.page_content)

                dict = {"file": file, "page no.": page, "excerpt": content}
                result = pd.DataFrame(dict)

                file_match_chunk_df = pd.concat(
                    [file_match_chunk_df, result], ignore_index=True
                )

                answer.append(chain.run(input_documents=matches, question=query))

        file_answer_df = pd.DataFrame({"file": file_name, "response": answer})
        text_to_summarize = "\n".join(answer)

        chat_messages = [
            SystemMessage(
                content=f"You are an academic research expert tasked to answer the following question: '{query}'"
            ),
            HumanMessage(
                content=f"answer the question by providing a single concise summary of the following text: '{text_to_summarize}'. \n Keep the summary within 1000 words "
            ),
        ]
        summary = summary_llm(chat_messages).content
        st.write("Summarised Answer:")
        st.write(summary)

        st.write("Response per file:")
        st.dataframe(file_answer_df)
        st.download_button(
            "Download csv",
            file_answer_df.to_csv(),
            file_name="file_answer.csv",
            mime="text/csv",
        )

        st.write("Matching portions from files:")
        st.dataframe(file_match_chunk_df)
        st.download_button(
            "Download csv",
            file_match_chunk_df.to_csv(),
            file_name="relevant_chunks.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
