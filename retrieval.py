import os
import openai
import sys
import streamlit as st

## api_key
api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if api_key:

    sys.path.append('../..')

    # from dotenv import load_dotenv, find_dotenv
    # _ = load_dotenv(find_dotenv()) # read local .env file

    #assign api key
    os.environ['OPENAI_API_KEY']  = api_key

    ## load vector store from stored database
    import chromadb
    from langchain_community.vectorstores import Chroma

    persist_directory = './No_splitter/chroma'
    ## embeddings
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
    from langchain.chat_models import ChatOpenAI
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


#--------------------------------------------


# streamlit framework
st.title('Coursera Courses !!!')
def generate_response(input_text):
    result = qa({"question": input_text})
    st.info(result['answer'])


with st.form('my_form'):
    text = st.text_area('Enter text:', 'hey what is your query?')
    submitted = st.form_submit_button('Submit')
    if not api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and api_key.startswith('sk-'):
        generate_response(text)
    