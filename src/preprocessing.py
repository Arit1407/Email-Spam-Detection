import re
import string
import nltk

from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r"http\S+|www\.\S+", "", text)

    text = re.sub(r'^subject\s*:\s*', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    text = " ".join(text.split())

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)