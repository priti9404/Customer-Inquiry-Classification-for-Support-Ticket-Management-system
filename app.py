from flask import Flask, render_template, request,  redirect, url_for, send_file
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download stopwords if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
app.config['FILE_UPLOADS'] = "uploads"
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# NLP Preprocessing Function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmatization and Stemming
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in lemmatized]
    
    return stemmed

# Function to train the model with uploaded CSV file
def train_model(file_path, x_column, y_column):
    # Load data from CSV file
    df = pd.read_csv(file_path)

    # Select the columns specified by the user
    X = df[x_column]
    y = df[y_column]

     # Apply NLP preprocessing
    X = X.apply(lambda text: ' '.join(preprocess_text(text)))

    # TF-IDF vectorization
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Initialize and train classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(clf, 'model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    # Model evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    coef = clf.coef_.tolist()  # Logistic Regression coefficients for each feature
    
     # Generate word cloud image
    generate_wordcloud(X)

    return accuracy, class_report, coef

# Word Cloud Generation Function
def generate_wordcloud(X):
    # Create a single string of all text
    all_text = ' '.join(X)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    
    # Plot and save word cloud as an image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('static/wordcloud.png')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    data = []
    message = ""
    accuracy = None
    class_report = None
    coef = None
    
    if request.method == 'POST':
        if 'filename' in request.files and request.form['x_column'] and request.form['y_column']:
            uploaded_file = request.files['filename']
            x_column = request.form['x_column']  # Get the input feature column name
            y_column = request.form['y_column']  # Get the target category column name
            
            file_path = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Train model with the uploaded file and specified columns
            accuracy, class_report, coef = train_model(file_path, x_column, y_column)
            message = "Model trained successfully!"

    return render_template('index.html',data=data, message=message,accuracy=accuracy, class_report=class_report, coef=coef)

if __name__ == '__main__':
    app.run(port=3000, debug=True)