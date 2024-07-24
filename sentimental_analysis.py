import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import sys

# Download VADER lexicon if not already available
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(comment):
    sentiment_score = sia.polarity_scores(comment)
    return sentiment_score['compound']

# Check if comments were provided as input
if len(sys.argv) != 2:
    print("Usage: python script_name.py <comments>")
    sys.exit(1)

comments = sys.argv[1]

# Proceed if comments were successfully read
if comments:
    # Create DataFrame from comments
    df = pd.DataFrame({'comments': [comments]})

    # Analyze sentiment for the comment
    df['sentiment_score'] = df['comments'].apply(analyze_sentiment)
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Negative' if x < -0.2 else 'Positive')

    # Print results
    print(df)

    # Report negative comments
    negative_comments = df[df['sentiment'] == 'Negative']
    # print("Negative Comments Report:")
    # print(negative_comments)
else:
    print("No comments to process.")
