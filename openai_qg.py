import os
import openai
import sys
import streamlit as st
import numpy as np
import pandas as pd
from openai import OpenAI
import wikipedia

## Function to generate a combined PDF from multiple QUESTIONS

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import utils
from io import BytesIO


class QG():
    def __init__(self,api_key) -> None:
        self.api_key = api_key
    def GEN_QUESTION(self):
        sys.path.append('../..')

        #assign api key
        os.environ['OPENAI_API_KEY']  = self.api_key

        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key = self.api_key
        )

            ## prompt
        # Update the prompt to instruct the model to generate questions from the given text.
        self.prompt = """
            Your task is to generate questions and its answers from the following text.
            Create questions that would be relevant to test a learner's understanding of the context. For example if the content
            given is related to Support Vector Machines, you can ask all questions related to this machine learning topic. It can be in different 
            difficulty levels (beginner, intermediate, advanced). Also give answers for corresponding questions. 
            Aim for a variety of questions. Create 10 multiple choice questions (MCQ), not less or more -exactly 10 MCQ questions- 
            provide each options in separate lines ('\n'). A sample output format is as follows: 
            Question number here e.g. 1): What is the main purpose of the k-nearest neighbors algorithm? a) Clustering data points b) Classifying data points c) Predicting continuous values d) Dimensionality reduction
            Answer: provide your answer here. After each answer provide two line breaks (that is '\n\n') so that each section of question and answer is evidently separated.
            Be stick to this output format.
            Text: {}
        """

        ## prompt for extracting a topic
        # Update the prompt to instruct the model to generate questions from the given text.
        self.prompt2 = """
            You have to be very very very specific in this extraction task. based on the text given, you 
            have to extract a single topic. It should not be a sentence. for example, text: I want to check my understanding on k-nearest neighbours, from
            this you have to extract topic as 'k-nearest neighbours machine learning'. so it has to be very very very specific and concise to the point.
            Text: {}
        """

        QUESTIONS = []
        combined_pdf = self.generate_combined_pdf(['No Questions yet yet'], ["no subtopics yet"])

        with st.form('my_form'):
            input_query = st.text_area("Check your Understanding ?? Enter your topic here: \U0001F600:", '')
            submitted = st.form_submit_button('Submit')
            input_topic = self.Extract_topic(input_query)
            if submitted and input_topic:
                
                subtopics = wikipedia.search(f'{input_topic}')
                subtopics = list(set(subtopics))  ## take unique subtopics
                for subtopic in subtopics:
                    
                    try:
                        summary = wikipedia.summary(subtopic, sentences = 100)
                        questions = self.QUESTION_GENERATION(summary)
                        st.write(f'Questions for subtopic: {subtopic}')
                        self.generate_response(questions)  ## display output
                        QUESTIONS.append(questions)  ## 
                    except:
                        print(f'exception for {subtopic}')
                        
                # Generate the combined PDF from the QUESTIONS
                combined_pdf = self.generate_combined_pdf(QUESTIONS, subtopics)
            
        # Provide a button to download the combined PDF file
        st.download_button(
            label="Download Combined Summary as PDF",
            data=combined_pdf.getvalue(),
            file_name="combined_summary.pdf",
            mime="application/pdf"
        )


    def chat_gpt(self,prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    


    # Define the QUESTION_GENERATION function
    def QUESTION_GENERATION(self,text):
        return self.chat_gpt(self.prompt.format(text))

    def generate_response(self,input_text):
        st.info(input_text)

    def Extract_topic(self,text):
        return self.chat_gpt(self.prompt2.format(text))
    
    ## pdf generator
    def generate_combined_pdf(self,QUESTIONS, subtopics):
    # Create a BytesIO object to hold the PDF in memory
        pdf_bytes = BytesIO()
        
        # Create a canvas object from the BytesIO object
        c = canvas.Canvas(pdf_bytes, pagesize=letter)
        
        # Define parameters for the layout
        left_margin = 1 * inch
        right_margin = 1 * inch
        top_margin = 1 * inch
        bottom_margin = 1 * inch
        line_height = 14
        page_width, page_height = letter
        
        # Iterate through each summary and add it to the PDF
        for i, summary in enumerate(QUESTIONS):
            # Add a title for each subtopic
            c.setFont("Helvetica-Bold", 14)
            title = f"Summary for: {subtopics[i]}"
            c.drawString(left_margin, page_height - top_margin, title)
            
            # Add the summary text to the PDF
            c.setFont("Helvetica", 12)
            
            # Define the current position for the summary text
            current_y = page_height - top_margin - line_height
            
            # Define the available width for wrapping
            available_width = page_width - left_margin - right_margin
            
            # Use reportlab's utility function for text wrapping
            wrapped_lines = utils.simpleSplit(summary, c._fontname, c._fontsize, available_width)
            
            # Iterate through each wrapped line
            for line in wrapped_lines:
                # Draw each line of the summary
                c.drawString(left_margin, current_y, line)
                
                # Move the current_y position
                current_y -= line_height
                
                # Check if we need to add a new page
                if current_y < bottom_margin:
                    c.showPage()
                    # Reset position for new page
                    current_y = page_height - top_margin
            
            # Add a new page for the next subtopic, if necessary
            if i < len(QUESTIONS) - 1:
                c.showPage()

        # Save the PDF data to the BytesIO object
        c.save()
        
        # Reset the cursor of the BytesIO object to the beginning
        pdf_bytes.seek(0)
        
        return pdf_bytes




