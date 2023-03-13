import string
import nltk
import streamlit as st
from urllib.request import urlopen
from bs4 import BeautifulSoup


nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


def preprocess_text(text):
    # Remove unnecessary characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("\n", " ").replace("\r", "")

    # Convert to lowercase
    text = text.lower()

    # Tokenize text into individual words
    words = nltk.word_tokenize(text)

    return words


def label_entities(words):
    ner = nltk.ne_chunk(nltk.pos_tag(words))
    return ner


def extract_features(ner):
    featuresets = []
    for i in range(len(ner)):
        if type(ner[i]) is nltk.tree.Tree:
            label = ner[i].label()
            word_list = [word for word, tag in ner[i].leaves()]
            for word in word_list:
                features = {"length": len(word),
                            "pos": nltk.pos_tag([word])[0][1],
                            "is_capitalized": word[0].upper() == word[0]}
                featuresets.append((features, label))

    return featuresets


def train_classifier(featuresets):
    # Split data into training and testing sets
    train_set, test_set = featuresets[:int(len(featuresets)*0.8)], featuresets[int(len(featuresets)*0.8):]

    # Train classifier on training set
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # Test classifier on testing set
    accuracy = nltk.classify.accuracy(classifier, test_set)
    return classifier, accuracy


def label_entities_with_classifier(text, classifier):
    entities = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, "label") and chunk.label() in classifier.labels():
                entities.append((chunk.label(), " ".join(c[0] for c in chunk.leaves())))

    return entities


def get_text_from_url(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    return text


def main():
    st.title("Historical Text Named Entity Recognition for Project Gutenberg")

    # Enter URL or upload file
    option = st.radio("Select Input Option", ("Enter URL", "Upload Text File"))

    if option == "Enter URL":
        url = st.text_input("Enter URL")
        if url:
            text = get_text_from_url(url)
        else:
            st.warning("Please enter a valid URL")
            return
    else:
        file = st.file_uploader("Upload Text File")
        if file is not None:
            text = file.read().decode("utf-8")
        else:
            st.warning("Please upload a text file")
            return

    # Preprocess text
    words = preprocess_text(text)

    # Label named entities
    ner = label_entities(words)

    # Extract features and train classifier
    featuresets = extract_features(ner)
    classifier, accuracy = train_classifier(featuresets)

    st.write(f"Accuracy: {accuracy:.2f}")

    # Label named entities using trained classifier
    entities = label_entities_with_classifier(text, classifier)

    # Display named entities
    st.write("Named Entities:")
    for entity in entities:
        st.write(f"{entity[0]}: {entity[1]}")


if __name__ == "__main__":
    main()
