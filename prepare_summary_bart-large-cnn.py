import streamlit as st
import numpy as np
import pandas as pd
# from tqdm import tqdm
# from tqdm.pandas import tqdm

# Initialize tqdm
# tqdm.pandas()

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("summarization", model="facebook/bart-large-cnn", max_length=60)

df = pd.read_csv("coursera_edx_main_features.csv")

def SUMMARY(x):
    if len(x)< 300:
        return x
    else:
        return pipe(x)

# print(SUMMARY(df['Decription'][16]))
df['Short Description'] = df['Decription'].apply(SUMMARY)
df.drop("Decription", axis=1, inplace=True)



df = df[['Title','Organization', 'Level', 'Ratings', 'Short Description', 'Skills', 'Completion Time', 'Instructor','Cost', 'Link']]
df.to_csv("coursera_edx_main_desc_summary.csv", index= False)