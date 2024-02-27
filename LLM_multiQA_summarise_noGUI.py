# ask single question
# print table of pdf name and answer , download button to download table as csv
# pass table list of answers to 2nd llm to summarise answers and print to screen

# in , future automate find metadata for papers and save to table for download
# generate reference list with options for style, downloadable as text file


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

key = ""

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)


llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    openai_api_key=key,
    # openai_api_key=st.session_state.openai_key,
)


chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=key,
    # openai_api_key=st.session_state.openai_key
)

import os

dir = "C:/Users/reu34/Desktop/eczema_LLMchatbot(GAIRYv4)/pdfs/"
pdf_docs = [dir + f for f in os.listdir(dir)]


file_name = []
answer = []

file_match_chunk_df = pd.DataFrame({"file": [], "page no.": [], "excerpt": []})

for pdf in pdf_docs:

    pdf_reader = PyPDFLoader(pdf)

    pages = pdf_reader.load_and_split(text_splitter=text_splitter)

    # pdf_reader = PdfReader("C:/Users/reu34/Desktop/eczema_LLMchatbot(GAIRYv4)/pdfs/test.pdf")
    # file_name = pdf.metadata.title
    # file_names.append(file_name)
    # for page in pdf_reader.pages:
    #     text += page.extract_text()
    # chunks = text_splitter.split_text(text)
    # vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    vectorstore = FAISS.from_documents(documents=pages, embedding=embeddings)

    matches = vectorstore.similarity_search(query, k=3)

    # alternate way to query vectorstore, but hidden similarity search step, cannot see chunk matches or control search parameters
    # from langchain.chains import RetrievalQA

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(),
    # )
    # result = qa.run(query)
    # print(result)

    file = []
    page = []
    content = []
    for match in matches:
        file.append(re.sub(r".*/", "", match.metadata["source"]))
        page.append(match.metadata["page"] + 1)
        content.append(match.page_content)
        # print(
        #     str(match.metadata["source"]) + str(match.metadata["page"] + 1) + ":",
        #     match.page_content[:300],
        # )
    dict = {"file": file, "page no.": page, "excerpt": content}

    file_match_chunk_df = pd.concat(
        [file_match_chunk_df, pd.DataFrame(dict)], ignore_index=True
    )
    file_name.append(re.sub(r".*/", "", pdf_reader.file_path))
    answer.append(
        chain.run(input_documents=matches, question=query)
    )  # return on summarised answer as string

    # to return dictionary containing input documents, question and output_text
    # answer = chain(
    #     {"input_documents": matches, "question": query}, return_only_outputs=True
    # )

    # from langchain.vectorstores import Chroma

    # vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity", search_kwargs={"k": 6}
    # )
    # retrieved_docs = retriever.invoke(query)
    # answer = chain.run(input_documents=retrieved_docs, question=query)

file_answer_df = pd.DataFrame({"file": file_name, "answer": answer})


# llm summarise answers from each document into a single summary

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

summary_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=key
)

summary = summary_llm(chat_messages).content

print(summary)
