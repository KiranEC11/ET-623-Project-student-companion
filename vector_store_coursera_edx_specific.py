import pandas as pd
import os
import openai
import sys

## api key
api_key =""

sys.path.append('../..')

#assign api key
os.environ['OPENAI_API_KEY']  = api_key

## load data

df = pd.read_csv("coursera_edx_specific.csv")
## create docs
from langchain_core.documents.base import Document

docs = [Document(page_content=content, metadata={"course_name": Title, "id": id}) for id, Title, content in df.itertuples(index=False)]

print(docs[0])
## text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

## NO SPLITTING


# r_spliiter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 100,
#     separators=["\n\n", "(?<=\. )", " ", "", "."]
# )

# DOCS = r_spliiter.split_documents(docs)

# create embeddings

from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key=api_key)

## chroma db
from langchain_community.vectorstores import Chroma

persist_directory = './coursera_edx_specific/chroma'

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)

#save this vector store
vectordb.persist()
print('vector store is saved')