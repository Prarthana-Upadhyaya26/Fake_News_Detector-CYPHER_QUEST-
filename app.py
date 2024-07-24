import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
from urllib.parse import urlparse
import subprocess
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

# Configure Google Generative AI
GEMINI_API_KEY = "GEMINI_API_KEY"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize Google Generative AI
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are specialized in Computer networks.",
)
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get domain name from URL
def get_domain_name(url):
    domain = urlparse(url).netloc
    return domain

# Function to get content from URL
def get_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error accessing {url}: {e}"

# Define PromptTemplate class
class PromptTemplate:
    def _init_(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)

# Streamlit App Configuration for "TruthShield AI Guard"
st.set_page_config(page_title="TruthShield AI Guard", page_icon="🛡", layout="wide")
st.title("TruthShield AI Guard")

# Function to run comparison subprocess
def run_comparison(post_input, content_variable):
    command = ['python', 'compare_credible.py', post_input, content_variable]
    st.write("Running subprocess for comparison...")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        subprocess_output = result.stdout.strip()
        st.write(f"Comparison output: {subprocess_output}")
        st.session_state.comparison_result = subprocess_output
    except Exception as e:
        st.error(f"Error running comparison subprocess: {e}")

# Function to analyze sentiment of comments
def analyze_sentiment(comment):
    sentiment_score = sia.polarity_scores(comment)
    return sentiment_score['compound']

# Function to run sentiment analysis
def run_sentiment_analysis(comments):
    try:
        df = pd.DataFrame(comments, columns=['comments'])
        df['sentiment_score'] = df['comments'].apply(analyze_sentiment)
        df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Negative' if x < -0.2 else 'Positive')

        negative_comments = df[df['sentiment'] == 'Negative']
        st.subheader("Negative Comments Report:")
        st.dataframe(negative_comments)

        if len(negative_comments) > 0:
            st.warning("Warning: There are sensitive comments that may require attention.")

    except Exception as e:
        st.error(f"Error running sentiment analysis: {e}")

# Main Streamlit App code for "TruthShield AI Guard"
def main():
    post_input = st.text_input("Write your post:", max_chars=10000)

    if st.button("Start Fact Checking"):
        if post_input:
            command = ['python', 'sample.py', post_input]
            st.write("Running subprocess for fact checking...")

            result = subprocess.run(command, capture_output=True, text=True)
            subprocess_output = result.stdout.strip()

            quest = """
            Separate the given text into two parts: the main content and a list of URLs. Format the output as follows:

            Content: <main content>

            URLs: <list of URLs>

            Text: {your_text}
            """
            
            prompt_template = PromptTemplate(input_variables=["your_text"], template=quest)
            prompt = prompt_template.format(your_text=subprocess_output)
            response = llm.invoke(prompt)

            content_split = response.split("URLs:")
            content = content_split[0].replace("Content:", "").strip()
            urls_section = content_split[1].strip() if len(content_split) > 1 else "No URLs found."

            content_variable = content
            
            st.header("URLs")
            st.write("Your information is backed by a strong credible and reliable source of information. Few of these domains containing related content to your post are:")
            st.write(urls_section)

            st.session_state.post_input = post_input
            st.session_state.content_variable = content_variable

    if "post_input" in st.session_state and "content_variable" in st.session_state:
        if st.button("Continue with comparison"):
            run_comparison(st.session_state.post_input, st.session_state.content_variable)

        if st.button("Check Sensitivity"):
            if "comparison_result" in st.session_state:
                run_sentiment_analysis([st.session_state.comparison_result])
            else:
                st.warning("Please perform comparison before checking sensitivity.")

    else:
        st.warning("Please write a post and start fact checking before continuing.")

if __name__ == "_main_":
    main()