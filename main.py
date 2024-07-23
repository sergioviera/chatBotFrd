import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


#from langchain.document_loaders import PyPDFDirectoryLoader

load_dotenv()
# 1. Load, chunk and index the contents of the blog to create a retriever.
#loader = PyPDFLoader("https://www.frsf.utn.edu.ar/images/mbeninte/SISTEMAS_Correlatividades_Plan_2023.pdf")
#    web_paths=("https://www.frlp.utn.edu.ar/contenido-utn/SistemasOrdenanza1877.pdf",
#               "https://www.frsf.utn.edu.ar/images/mbeninte/SISTEMAS_Correlatividades_Plan_2023.pdf"),
#)
#docs = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#splits = text_splitter.split_documents(docs)

loader = PyPDFLoader("https://www.frlp.utn.edu.ar/contenido-utn/SistemasOrdenanza1877.pdf")
#    web_paths=("https://www.frlp.utn.edu.ar/contenido-utn/SistemasOrdenanza1877.pdf",
#               "https://www.frsf.utn.edu.ar/images/mbeninte/SISTEMAS_Correlatividades_Plan_2023.pdf"),
#)
docs = loader.load()
#print(len(docs))
#loader = PyPDFDirectoryLoader("/resoluciones/")
#docs= loader.load()
#print(len(docs))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs) 

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. You must respond only about the Information Systems Engineering program at UTN. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use a maximum of three sentences and keep the answer concise. Respond in Spanish."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

while(True):
    prompt=input()
    if prompt:
        response = rag_chain.invoke({"input": prompt})
        print(response["answer"])
        print("--")
