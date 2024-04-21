import pandas as pd
import os
import openai
import sys
import numpy as np

## api key
api_key =""

sys.path.append('../..')

#assign api key
os.environ['OPENAI_API_KEY']  = api_key


#-----------------------------------------------------------------------#
from openai import OpenAI
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# embd1 = get_embedding("hi this is kiran from India")
# embd2 = get_embedding("hi I am kiran from Japan")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


#-----------------------------------------------------------------------#
## load vector store from stored database
import chromadb
from langchain_community.vectorstores import Chroma

persist_directory = './No_splitter/chroma'

from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)
vectordb.get()

## query database

# query it
query = "courses which help to acquire sql and data interpretation skills"
docs = vectordb.similarity_search(query, k= 3)  ## normal search   --> the courses can repeat

results = vectordb.similarity_search_with_score(query, k=3)
# docs = []
# scores = []
# for item in results:
#     docs.append(item[0])
#     scores.append(item[1])
# print(scores)
# docs = vectordb.max_marginal_relevance_search(query, k=3)   ## diverse answers

similar_courses = []
similar_courses_content = []
for item in docs:
    similar_courses.append(item.metadata['course_name'])
    similar_courses_content.append(item.page_content)
print(f"similar courses :{similar_courses}")

courses_uniq = []
for x in similar_courses:
    if x not in courses_uniq:
        courses_uniq.append(x)

courses_uniq_content = []
for x in similar_courses_content:
    if x not in courses_uniq_content:
        courses_uniq_content.append(x)

print('---------\nUnique courses',courses_uniq)

query_embd = get_embedding(query)

## print cosine similarity
for i in range(len(courses_uniq_content)):
    print(cosine_similarity(query_embd, get_embedding(courses_uniq_content[i])))

print("similarity between two of the above courses",cosine_similarity(get_embedding(courses_uniq_content[0]), get_embedding(courses_uniq_content[1])))


# print('---------------------------------',vectordb.similarity_search(f"provide me url for {courses_uniq[0]} course")[0])  ### this simply retrieve the chunk which is very similar

## RETRIEVALQA
from langchain.chains import RetrievalQA
## chat openAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key=api_key  ,model_name="gpt-3.5-turbo", temperature=0)
retrievalQA = RetrievalQA.from_llm(
                                    llm=llm, 
                                    retriever=vectordb.as_retriever(),
                                    # return_source_documents= True
                                    )


# question = f"provide me url for {courses_uniq[0]} course"
# print(retrievalQA.invoke({"query": question})['result'])

#--------------------------------------------------------------------------#
"""
compression: deal with irrelevant data
"""
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")  ## compression + mmr
)

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.metadata['course_name'] for i, d in enumerate(docs)]))



compressed_docs = compression_retriever.get_relevant_documents(query)
# pretty_print_docs(compressed_docs)
print(f"compressed docs: {compressed_docs}")
courses__compression = [d.metadata['course_name'] for d in compressed_docs]
similar_courses_compression = []
similar_courses_compression_content = []
similar_courses_compression_content_filter = []
for item in compressed_docs:
    similar_courses_compression.append(item.metadata['course_name'])
    similar_courses_compression_content.append(item.page_content)


# for i in range(len(similar_courses_compression_content)):
#     print(f"similarity between {i}-th course and query is {cosine_similarity(get_embedding(similar_courses_compression_content[i]), query_embd)}")
#     for j in range(i,len(similar_courses_compression_content)):
#         print(f"similarity between {i} and {j} courses {cosine_similarity(get_embedding(similar_courses_compression_content[i]), get_embedding(similar_courses_compression_content[j]))}")

db_filtered = []
for item in similar_courses_compression:
    db_filtered.append(vectordb.get(where = {"course_name": item})['ids'][0])

# for item in db_filtered:
    # print(db_filtered)