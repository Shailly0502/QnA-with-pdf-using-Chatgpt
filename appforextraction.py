from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pandas as pd

os.environ["OPENAI_API_KEY"] = #your key
fname=input("Enter File name")
pdfreader = PdfReader(str(fname))

# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

#print(raw_text)

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
#print(texts)

embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

print("Enter 1 if you would like to search the document. \n Enter 0 if you would like to exit.")
will=int(input())
while(will!=0):
    query = input("Enter Query.")
    docs = document_search.similarity_search(query)
    txt=chain.run(input_documents=docs, question=query)
    print(txt)
    print("Enter 1 if you would like to search for different query. \n Enter 2 if you would like to add it to the database. \n Enter 0 if you would like to exit.")
    will=int(input())
    if(will==1):
        continue
    elif(will==2):
        print("Enter name of heading.")
        heading=input()
        listofkeys=txt.split("," or ";")
        df=pd.DataFrame(listofkeys,columns =[heading])
        df2=pd.read_csv("iso.csv")
        df = df.assign(ID=range(len(df)))
        df1 = pd.merge(df,df2, how='outer', on = 'ID', suffixes = ('_left', '_right'))
        df1.to_csv("iso.csv",index=False)
    elif(will==0):
        continue
print("Exit")