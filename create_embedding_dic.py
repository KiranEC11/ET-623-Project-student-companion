import os
import openai
import sys
import streamlit as st
import numpy as np

## api_key

api_key = ""

sys.path.append('../..')

#assign api key
os.environ['OPENAI_API_KEY']  = api_key

from openai import OpenAI
client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# ## load id2title, id2SpecificContent, id2MasterContent
import json

## id2MasterContent
json_file_path = "id2MasterContent.json"
with open(json_file_path, 'r') as json_file:
    id2MasterContent = json.load(json_file)


## id2SpecificContent
json_file_path = "id2SpecificContent.json"
with open(json_file_path, 'r') as json_file:
    id2SpecificContent = json.load(json_file)

## id2title
json_file_path = "id2title.json"
with open(json_file_path, 'r') as json_file:
    id2title = json.load(json_file)


id2MasterEmbedding = {}  ## dictionary with id as key and embedding as val --> this embeddings contains all the content 
id2SpecificEmbedding = {}  ## dictionary with id as key and embedding as val --> this embeddings contains only specific content (desc + skills + Filters)

# id2MasterEmbedding
for id, content in id2MasterContent.items():
    id2MasterEmbedding[id] = get_embedding(content)

# id2SpecificEmbedding
for id, content in id2SpecificContent.items():
    id2SpecificEmbedding[id] = get_embedding(content)

## save embeddings
json_file_path = 'id2MasterEmbedding.json'
with open(json_file_path, 'w') as json_file:
    json.dump(id2MasterEmbedding, json_file, indent=4)


json_file_path = 'id2SpecificEmbedding.json'
with open(json_file_path, 'w') as json_file:
    json.dump(id2SpecificEmbedding, json_file, indent=4)