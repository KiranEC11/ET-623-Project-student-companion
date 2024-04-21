import os
import openai
import sys
import streamlit as st
import numpy as np
import json
import pandas as pd
import math

## qg module
from openai_qg import QG

## load df
df = pd.read_csv("title,n_enrolled_normalised,ratings_normalised,ratio(n_rating,n_enrolled),id.csv")
df_main = pd.read_csv("comparison_ouput_short_desc_bart_large_modified.csv")

api_key = st.sidebar.text_input('OpenAI API Key', type='password')
is_content_rec = st.checkbox('Course recommender', help='Content based course recommender')
is_chatbot = st.checkbox('EduJARVIS', help= "ask chatbot your doubts regarding courses")
is_qg = st.checkbox('Am I good enough..?', help="If you want to test your knowledge on any particular topic, click here..!!")

if api_key and is_chatbot:

    sys.path.append('../..')

 

    #assign api key
    os.environ['OPENAI_API_KEY']  = api_key

    ## load vector store from stored database
    import chromadb
    from langchain_community.vectorstores import Chroma

    persist_directory = './coursera_edx_master/chroma'
    ## embeddngs
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    vectordb.get()


    ## LLM name
    import datetime
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    # print(llm_name)

    ## chat openAI
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(openai_api_key=api_key  ,model_name=llm_name, temperature=0)
    # llm.predict("Hello world!")

    # Memory
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # converstaional retrieval chains
    from langchain.chains import ConversationalRetrievalChain
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )


    st.title("EduJARVIS")
    st.write("EduJARVIS is a student companion. You can ask EduJARVIS your queries related to courses to learn, skills to acquire for your career path, what are popular courses etc.")
    def generate_response(input_text):
        result = qa({"question": input_text})
        st.info(result['answer'])


    with st.form('my_form'):
        text = st.text_area("Hey what's your query..?\U0001F600:", '')
        submitted = st.form_submit_button('Submit')
        if not api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and api_key.startswith('sk-'):
            generate_response(text)
elif api_key and is_content_rec:
    ## create some essential dictionaries
    id2normalised_rating = {}
    id2_normalised_n_enrolled  = {}
    id2_chi_ratio = {}
    for _, n_enrolled_normalised, ratings_normalised, chi_ratio, id in df.itertuples(index=False):
        id2_normalised_n_enrolled[id] = n_enrolled_normalised
        id2normalised_rating[id] = ratings_normalised
        id2_chi_ratio[id] = chi_ratio

    ## load embeddings json
                        
    ## specific embedding
    json_file_path = "id2SpecificEmbedding.json"
    with open(json_file_path, 'r') as json_file:
        id2SpecificEmbedding = json.load(json_file)

    ## master embedding
    json_file_path = "id2MasterEmbedding.json"
    with open(json_file_path, 'r') as json_file:
        id2MasterEmbedding = json.load(json_file)
    
    ## id2SpecificContent
    json_file_path = "id2SpecificContent.json"
    with open(json_file_path, 'r') as json_file:
        id2SpecificContent = json.load(json_file)
    
    ## id2title
    json_file_path = "id2title.json"
    with open(json_file_path, 'r') as json_file:
        id2title = json.load(json_file)


    sys.path.append('../..')

    #assign api key
    os.environ['OPENAI_API_KEY']  = api_key

    from openai import OpenAI
    client = OpenAI()


    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    ## load vector store from stored database
    import chromadb
    from langchain_community.vectorstores import Chroma

    ## load vectordb_master
    persist_directory = './coursera_edx_master/chroma'

    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    vectordb.get()

    ## load vectordb_specific
    persist_directory_specific = './coursera_edx_specific/chroma'

    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()

    vectordb_specific = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory_specific
    )
    vectordb_specific.get()

    ##
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(openai_api_key=api_key  ,model_name="gpt-3.5-turbo", temperature=0)
 
    st.title("Content based Course Recommender")
    def generate_response(input_text):
        st.info(input_text)


    with st.form('my_form'):
        text = st.text_area('Enter your query in your own words. Mention either topic name or skills to acquire:', '')
        level  = st.radio("pick your preferred difficulty level", ["any", "Introductory","Beginner", "Intermediate", "Advanced", "Mixed"])
        duration  = st.selectbox("pick maximum duration of the course in weeks", ["1", "5", "8" ,"10", "12", "20","30"])
        price = st.slider("pick maximum price in Rs.", 0,10000)
        query = text + f". The difficulty level is {level}. " + f"The duration of the course should be less than or equal to {duration} weeks. " + f"The price of the course should be less than Rs.{price}."
        # print(query)
        submitted = st.form_submit_button('Submit')
        if not api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and api_key.startswith('sk-'):
            ## get most similar but diverse courses to given query
            ## mmr retrieval
            retrieved_docs = vectordb.max_marginal_relevance_search(query, k=5)
            # print(compressed_docs)
            courses_ids_first_all = [d.metadata['id'] for d in retrieved_docs] ## first stage
            courses_ids_first = list(set(courses_ids_first_all))  ## select unique courses

            ## find similarity score between query and courses_ids_first
            query_embd = get_embedding(query)

            # stage_1 = [cosine_similarity(query_embd,x) for id in courses_ids_first for x in id2MasterEmbedding[id]]
            id2S1 = {}  ## stage 1 similarity
            for id in courses_ids_first:

                id2S1[id] = cosine_similarity(query_embd,id2MasterEmbedding[str(id)])

            id2S2 = {}   ## stage 2 similarity (sim1 * sim2)
            id2SQ = {}
            
            all_courses = []  ## collection of all the courses
            for id, similarity in id2S1.items():
                query_content = id2SpecificContent[str(id)]
                retrieved_docs_2 = vectordb_specific.similarity_search(query_content, k=5)    ## retrieve 5 similar courses
                courses_id_second_all = [d.metadata['id'] for d in retrieved_docs_2]  ## second stage
                
                courses_id_second_all_new = [x for x in courses_id_second_all if x not in all_courses and x not in id2S1.keys()]  ## take only those courses which are not in courses_ids_first
                all_courses.extend(courses_id_second_all_new)
                
                for id_2 in courses_id_second_all_new:
                    id2S2[id_2] = cosine_similarity(id2SpecificEmbedding[str(id_2)],id2SpecificEmbedding[str(id)]) * similarity
                               
            id2similiarity = {**id2S1, **id2S2} ## combine id2S1 and id2S2
            
            for id in id2similiarity.keys():
                id2SQ[id] = cosine_similarity(query_embd, id2MasterEmbedding[str(id)])
            

            ## find final score for each ids
            id2Score = {}  ##  store score corr to each id
            for id in id2similiarity.keys():
                wr = math.log(1 + id2_chi_ratio[id])  # weight for rating
                score = (0.7*id2SQ[id]) + (0.3*id2similiarity[id]) + (wr*id2normalised_rating[id]) + (0.5*id2_normalised_n_enrolled[id])
                id2Score[id] = round(score,5)
            
            # Sort the dictionary items based on values
            # The sorted function returns a list of tuples (key, value)
            sorted_items = sorted(id2Score.items(), key=lambda item: item[1], reverse=True)  ## sorting in descending order

            # Create a new dictionary from the sorted items
            id2Score_sorted = dict(sorted_items)
            ids_sorted = [id for id in id2Score_sorted.keys()]  ## ids in order
            best_similar = id2title[str(ids_sorted[0])]
            text_out1 = f"""
            The most similar course is {best_similar} by organization called {df_main['Organization'][ids_sorted[0]]}. Other similar courses are given below. Please have a look. Happy learning \U0001F600
            """
            generate_response(text_out1) 
            ids_sorted_mod = [x for x in ids_sorted if x in df_main.index]
            st.dataframe(df_main.loc[ids_sorted_mod], 10000,300)  ## display details of related most similar courses
elif is_qg and api_key:
    client_qg = QG(api_key)
    client_qg.GEN_QUESTION()
elif not api_key:
    st.warning('Please enter your OpenAI API key!', icon='⚠')
else:
    st.warning('Select an option from checkbox. Do not select multiple options.', icon='⚠')

bools = [is_content_rec, is_chatbot, is_qg]
if sum(bools)>1:
    st.warning('Do not select multiple options.', icon='⚠')
    
    

