import os
import streamlit as st
import pandas as pd
import subprocess
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain_google_genai import GoogleGenerativeAI  # Assuming correct import path
from googletrans import Translator  # Import googletrans library

# Configure Google Generative AI
GEMINI_API_KEY = "AIzaSyBxsNgKI-TQbKVi8qZyij9RICZJCqgOpbw"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

generation_config = {
    "temperature": 0.2,   # Lower temperature for more conservative and precise outputs
    "top_p": 1.0,         # Set top_p to 1.0 for deterministic outputs based on probability
    "top_k": 0,           # Disable top_k sampling for strict adherence to probabilities
    "max_output_tokens": 8192,   # Maximum number of output tokens to generate
    "response_mime_type": "text/plain",  # Output format type
}


# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize Google Translator
translator = Translator()

# Define PromptTemplate class
class PromptTemplate:
    def __init__(self, input_variables, template):
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
    st.write("Running subprocess2...")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        subprocess_output = result.stdout.strip()
        st.write(f"Subprocess output: {subprocess_output}")
        st.session_state.comparison_result = subprocess_output  # Store comparison result in session state
    except Exception as e:
        st.error(f"Error running subprocess: {e}")

# Function to analyze sentiment of comments
def analyze_sentiment(comment):
    sentiment_score = sia.polarity_scores(comment)
    return sentiment_score['compound']

# Function to run sentiment analysis
def run_sentiment_analysis(comments):
    try:
        # Create DataFrame from comments
        df = pd.DataFrame({'comments': comments})

        # Analyze sentiment for each comment
        df['sentiment_score'] = df['comments'].apply(analyze_sentiment)
        df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Negative' if x < -0.2 else 'Positive')

        # Report negative comments
        negative_comments = df[df['sentiment'] == 'Negative']
        st.subheader("Negative Comments Report:")
        st.dataframe(negative_comments)

        # Display warning for sensitive content
        if len(negative_comments) > 0:
            st.warning("Warning: There are sensitive comments that may require attention.")

    except Exception as e:
        st.error(f"Error running sentiment analysis: {e}")

# Function to translate text to English
def translate_to_english(text):
    translated = translator.translate(text, dest='en')
    return translated.text


# Main Streamlit App code for "TruthShield AI Guard"
def main():
    post_input = st.text_input("Write your post:", max_chars=10000)

    if st.button("Start Fact Checking"):
        if post_input:
            # Translate the input text to English
            post_input_english = translate_to_english(post_input)
            st.write(f"Translated text to English: {post_input_english}")

            command = ['python', 'research_llm.py', post_input_english]
            st.write("Running subprocess1...")

            # Run the subprocess and capture the output
            result = subprocess.run(command, capture_output=True, text=True)
            subprocess_output = result.stdout.strip()
            st.write(f"Subprocess1 output: {subprocess_output}")

            # Define the prompt template
            quest = """
            Separate the given text into two parts: the main content and a list of URLs. Format the output as follows:

            Content: <main content>

            URLs: <list of URLs>

            Text: {your_text}
            """
            prompt_template = PromptTemplate(input_variables=["your_text"], template=quest)
            prompt = prompt_template.format(your_text=subprocess_output)
            st.write(f"Generated prompt: {prompt}")

            # Assuming llm is initialized correctly somewhere
            llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
            response = llm.invoke(prompt)

            # Split the generated text into content and URLs
            content_split = response.split("URLs:")
            content = content_split[0].replace("Content:", "").strip()
            urls_section = content_split[1].strip() if len(content_split) > 1 else "No URLs found."

            # Store content in a variable
            content_variable = content
            st.write(f"Extracted content: {content_variable}")

            st.header("URLs")
            st.write("Your information is backed by a strong credible and reliable source of information. Few of these domains containing related content to your post are:")
            st.write(urls_section)

            # Store state to prevent page reload
            st.session_state.post_input = post_input_english
            st.session_state.content_variable = content_variable

    # Check if post_input and content_variable are in session state to continue
    if hasattr(st, 'session_state') and "post_input" in st.session_state and "content_variable" in st.session_state:
        if st.button("Continue with comparison"):
            run_comparison(st.session_state.post_input, st.session_state.content_variable)

        if st.button("Check Sensitivity"):
            if "comparison_result" in st.session_state:
                run_sentiment_analysis([st.session_state.comparison_result])
                command = ['python', 'sentimental_analysis.py', st.session_state.post_input]
                st.write("Running subprocess3...")

                # Run the subprocess and capture the output
                result = subprocess.run(command, capture_output=True, text=True)
                subprocess_output = result.stdout.strip()
                st.write(f"Subprocess3 output: {subprocess_output}")

            else:
                st.warning("Please perform comparison before checking sensitivity.")

    else:
        st.warning("Please write a post and start fact checking before continuing.")

if __name__ == "__main__":
    main()
