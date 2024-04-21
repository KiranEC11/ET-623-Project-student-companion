import os
import openai
import sys
import streamlit as st
import numpy as np
import pandas as pd

## api_key
api_key = ""


sys.path.append('../..')

#assign api key
os.environ['OPENAI_API_KEY']  = api_key

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key = "",
)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()



prompt ="""
    Your task is to extract relevant information from a text which is description about a course. 
    This information is used to create a summary of course description.
    Extract relevant information from the following text.
    Be sure to preserve the important details.
    Text: {}"""

# print(chat_gpt(prompt.format("Understand the value of data")))

def SUMMARY(x):
    return chat_gpt(prompt.format(x))

df = pd.read_csv("coursera_edx_main_features.csv")
print("data loaded..!!")
df['Short Description'] = df['Decription'].apply(SUMMARY)
df.drop("Decription", axis=1, inplace=True)



df = df[['Title','Organization', 'Level', 'Ratings', 'Short Description', 'Skills', 'Completion Time', 'Instructor','Cost', 'Link']]
df.to_csv("coursera_edx_main_desc_summary.csv", index= False)
print("summariser completed")