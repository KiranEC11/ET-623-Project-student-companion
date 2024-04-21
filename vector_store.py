import pandas as pd
import os
import openai
import sys

## api key
api_key =""

sys.path.append('../..')

#assign api key
os.environ['OPENAI_API_KEY']  = api_key


## load df
df  = pd.read_csv("Coursera.csv")

## template
template = 'Course name is {}. This course is available in coursera education platform. The course is provided by the {}. The course description is as follows:\n {}\n. The student can acquire the following skills after completion of this course: the skils are {}\n. The level of difficulty for this course is {}. Currently the course has a rating of {}. You can find the course from the {} this url.'


courses_list = [
    template.format(
        name, university, description, skills, level, rating, url
    )
    
    for name, university, level, rating, url, description, skills in df.itertuples(index=False)
]


# ## save data in template
# file_path = "Coursera_courses.txt"


# with open(file_path, "w") as file:
#     for i in range(len(courses_list)):
#         file.write(courses_list[i])

##  vector store

## create docs
from langchain_core.documents.base import Document

docs = [Document(page_content=content, metadata={"course_name":name}) for content, name in zip(courses_list, df['Course Name'].values)]



## text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# r_spliiter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 100,
#     separators=["\n\n", "(?<=\. )", " ", ""]
# )

# DOCS = r_spliiter.split_documents(docs)

# create embeddings

from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key=api_key)

## chroma db
from langchain_community.vectorstores import Chroma
persist_directory = './No_splitter/chroma'

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)

#save this vector store
vectordb.persist()
print('vector store is saved')