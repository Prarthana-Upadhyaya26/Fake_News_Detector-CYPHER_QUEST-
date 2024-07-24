import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# # Define file paths
# source_file_path = r"C:\Users\prart\OneDrive\Desktop\source.txt"
# input_file_path = r"C:\Users\prart\OneDrive\Desktop\input.txt"

# # Function to read and clean text
# def read_and_clean_text(file_path):
#     try:
#         with open(file_path, 'r') as f:
#             text = f.read()
#             # Remove special characters, extra spaces, and line breaks
#             text = re.sub(r'\s+', ' ', text)
#             text = re.sub(r'[^\w\s]', '', text)
#             return text
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         return ""

# Reading and cleaning source and input texts
if len(sys.argv) != 3:
        print("Usage: python other_script.py <source_text> <input_text>")
        sys.exit(1)
    
source_text = sys.argv[1]
input_text = sys.argv[2]

# Tokenization
source_tokens = word_tokenize(source_text)
input_tokens = word_tokenize(input_text)

# Removing stopwords
stop_words = set(stopwords.words('english'))
source_tokens = [word for word in source_tokens if word.lower() not in stop_words]
input_tokens = [word for word in input_tokens if word.lower() not in stop_words]

# Optional: Stemming
stemmer = PorterStemmer()
source_tokens = [stemmer.stem(word) for word in source_tokens]
input_tokens = [stemmer.stem(word) for word in input_tokens]

print("Source tokens:", source_tokens)
print("Input tokens:", input_tokens)

# Combine tokens back to strings for vectorization
source_text_processed = ' '.join(source_tokens)
input_text_processed = ' '.join(input_tokens)

# Check if input text is a substring of source text
def is_substring(source_text, input_text):
    return input_text.lower() in source_text.lower()

substring_match = is_substring(source_text, input_text)

if substring_match:
    credibility_percentage = 98
else:
    # Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([source_text_processed, input_text_processed])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    credibility_percentage = similarity[0][0] * 100

print("Credibility Percentage:", credibility_percentage)

if(credibility_percentage <= 35):
    print("Warning: It's a personal opnionated post, the information in post can be human speculations and not factual information.")
elif(credibility_percentage <= 75):
    print("The post has enough verified content based on factual information, you are good to post.")
else:
    print("The post is too good to be true, there might be a copyright issue. ")