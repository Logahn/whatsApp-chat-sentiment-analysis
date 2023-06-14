from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji

from collections import Counter
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

"""## Definition of Functions"""


import re

def date_time(date_string: str) -> bool:
    """
    Validates if a string is a valid date and time format.

    Args:
        date_string (str): The string to be validated.

    Returns:
        bool: True if the string is a valid date and time format, False otherwise.
    """
    # The regular expression pattern to match against the date string
    pattern = r'^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'

    # Attempt to match the pattern against the date string
    match = re.match(pattern, date_string)

    # Return True if there was a match, False otherwise
    return bool(match)

# Contact extraction


def find_contact(string: str) -> bool:
    """
    This function takes a string as input and returns True if the string has exactly one colon (':') character, otherwise False.
    """
    # Split the string by colon (':') character
    split_string = string.split(":")
    
    # Check if the string has exactly one colon
    if len(split_string) == 2:
        return True
    else:
        return False

# Message extraction


def getMessage(line):
    splitline = line.split(" - ")
    datetime = splitline[0]
    date, time = datetime.split(", ")
    message = " ".join(splitline[1:])

    if find_contact(message):
        splitmessage = message.split(": ")
        author = splitmessage[0]
        message = " ".join(splitline[1:])
    else:
        author = None
    return date, time, author, message


"""### Data pre-processing"""

data = []
conversation = 'sample chat.txt'
with open(conversation, encoding="utf-8") as fp:
    fp.readline()
    messageBuffer = []
    date, time, author = None, None, None
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()

        if date_time(line):
            if len(messageBuffer) > 0:
                data.append([date, time, author, ''.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = getMessage(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)

"""### Sentiment analysis"""

nltk.download('vader_lexicon')
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.DataFrame(data, columns=["Date", "Time", "Contact", "Message"])
df['Date'] = pd.to_datetime(df['Date'])
data = df.dropna()

sentiments = SentimentIntensityAnalyzer()

data["positive"] = [sentiments.polarity_scores(
    i)["pos"] for i in data["Message"]]
data["negative"] = [sentiments.polarity_scores(
    i)["neg"] for i in data["Message"]]
data["neutral"] = [sentiments.polarity_scores(
    i)["neu"] for i in data["Message"]]

data

x = sum(data["positive"])
y = sum(data["negative"])
z = sum(data["neutral"])


def score(pos, neg, neu):
    if pos > neg and pos > neu:
        print("Positive")
    elif neg > pos and neg > neu:
        print("Negative")
    else:
        print("Neutral")


score(x, y, z)

fig = np.array([x, y, z])
lab = ["Positive", "Negative", "Neutral"]
mycolors = ["green", "red", "orange"]
myexplode = [0.2, 0, 0]


plt.pie(fig, labels=lab, colors=mycolors, explode=myexplode)
plt.show()
