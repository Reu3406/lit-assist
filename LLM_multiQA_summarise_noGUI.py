'''
what the python script does 
# you direct python to a folder of pdf files
# you ask a single question to all the pdf files
# each pdf file is chunked into pieces and vectorised by embedding model , chunks stored into a vector database
# your question is matched to the chunks of each pdf , through similarity search, the top few matching chunks are produced 
# from the matching chunks of each document , the LLM summarises an answer to your question for each pdf
# the LLM then summarises all answers for each pdf into a single final answer to your question
# the table of pdf name and LLM's answer for each pdf, and the table of each pdf and the matching chunks of text (with page nunmber) to the question is stored and saved for reference 
'''

'''
if you don't have the langchain package already installed :
pip install --upgrade langchain 
'''

#from PyPDF2 import PdfReader  #only needed for optional alternative methods commented out below
#import pypdf
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

#input your OpenAI account API key 
key = ""

#defining text splitter to split text read from pdf files
#splitter will split by special characters first , i.e by \n next line first...etc
#then it will split by chunk size, in this case 500 character lengths , with overlap of 100 characters between chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

#defining LLM that will be used to summarise answer from matching chunks of text from each pdf
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    openai_api_key=key,
    # openai_api_key=st.session_state.openai_key,
)

#the command to QA(query-answer) LLM using the LLM parameters defined above
chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
)

#defining embedding model that will be used to vectorise text to embedding vectors that can be understood by LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=key,
    # openai_api_key=st.session_state.openai_key
)

#for getting path address of files
import os

#input full address of pdf folder
dir = "C:/Users/reu34/Desktop/eczema_LLMchatbot(GAIRYv4)/pdfs/"

#lists all pdf files in the folder, and append the full address path to each file name, so pdf reader can find and read files later
pdf_docs = [dir + f for f in os.listdir(dir) if f.endswith(".pdf")]

#creating empty lists to fill in later 
file_name = []
answer = []
#create empty dataframe from dictionary of column name:content list, for later fill in 
file_match_chunk_df = pd.DataFrame({"file": [], "page no.": [], "excerpt": []})

#loop through each pdf file in pdf folder
for pdf in pdf_docs:
    
    pdf_reader = PyPDFLoader(pdf) #read in pdf as documents with page numbers

    pages = pdf_reader.load_and_split(text_splitter=text_splitter) #split text using text splitter, retains page number reference

    #alternate way to read in pdf and store as vector database, except this way no page number, every page is read and appended together into single long string of text before splitting
    # pdf_reader = PdfReader("C:/Users/reu34/Desktop/eczema_LLMchatbot(GAIRYv4)/pdfs/test.pdf")
    # file_name = pdf.metadata.title
    # file_names.append(file_name)
    # for page in pdf_reader.pages:
    #     text += page.extract_text()
    # chunks = text_splitter.split_text(text)
    # vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    #from split documents , convert to embedding vectors using embedding model and store as vector database
    vectorstore = FAISS.from_documents(documents=pages, embedding=embeddings)

    #similarity search of question against chunks of text , k=3 to produce top 3 matches 
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

    #create empty lists for file name, page number and text content of matches to be filled in
    file = []
    page = []
    content = []
    #loop through each matching chunk from similarity search
    for match in matches:
        #fill in file name , page number and content , and append to respective lists
        file.append(re.sub(r".*/", "", match.metadata["source"]))
        page.append(match.metadata["page"] + 1)
        content.append(match.page_content)
    #create dictionary from the lists of file name, page numnber and content, and use dictionary to append to dataframe
    dict = {"file": file, "page no.": page, "excerpt": content}
    file_match_chunk_df = pd.concat(
        [file_match_chunk_df, pd.DataFrame(dict)], ignore_index=True
    )
    #append file name and summarised answer from chunks to respective lists
    file_name.append(re.sub(r".*/", "", pdf_reader.file_path))
    answer.append(chain.run(input_documents=matches, question=query))  # returns only summarised answer as string

    # to return entire dictionary containing input documents, question and output_text
    # answer = chain({"input_documents": matches, "question": query}, return_only_outputs=True)

    #alternate way using ChromaDB instead of FAISS vectordatabase to similarity search and QA LLM
    # from langchain.vectorstores import Chroma
    # vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity", search_kwargs={"k": 6}
    # )
    # retrieved_docs = retriever.invoke(query)
    # answer = chain.run(input_documents=retrieved_docs, question=query)

#use the lists of file name and summarised answer to create dataframe 
file_answer_df = pd.DataFrame({"file": file_name, "answer": answer})


# summarise answers from each document into a single summarised answer
#append all answers from each pdf into a long string
text_to_summarize = "\n".join(answer)
# prepare prompts to LLM to summarize all answers into final answer
#using f strind to input question and long string of text to be summarised, into the prompt messages
chat_messages = [
    SystemMessage(
        content=f"You are an academic research expert tasked to answer the following question: '{query}'"
    ),
    HumanMessage(
        content=f"answer the question by providing a single concise summary of the following text: '{text_to_summarize}'. \n Keep the summary within 1000 words "
    ),
]


#defining and calling LLM to summarise answers into final answer
summary_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=key
)

summary = summary_llm(chat_messages).content

print(summary)


# saving csv files of matching chunks and each answer to same folder as pdfs, using f string to input the folder address same as the pdf documents
file_answer_df.to_csv(f"{dir}LLMresponse_perfile.csv")
file_match_chunk_df.to_csv(f"{dir}matching_portions_of_text.csv")



