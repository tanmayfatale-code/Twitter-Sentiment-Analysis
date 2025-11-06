# 🐦 Twitter Sentiment Analysis: ML Classification

## Project 3: Data Science Portfolio Project

This project implements a robust sentiment analysis pipeline for classifying tweet text as *Positive, Negative, or Neutral*. It transitions from a simple lexicon-based approach (TextBlob) to a trained Machine Learning model for improved accuracy and scalability.

---

## 🚀 Key Technologies & Skills

* *Language:* Python
* *Data Handling:* pandas (DataFrames)
* *Natural Language Processing (NLP):* Regular Expressions (re), Text Cleaning, *TF-IDF Vectorization. * **Machine Learning:* scikit-learn (*Logistic Regression Classifier*)
* *Model Persistence:* joblib (Saving the trained model and vectorizer).
* *Visualization:* matplotlib (Pie Charts)

## ⚙ ML Pipeline Overview

1.  *Data Preprocessing:* Raw tweets are cleaned (removing URLs, mentions, hashtags).
2.  *Feature Engineering:* Cleaned text is converted into numerical features using *TF-IDF*.
3.  *Model Training:* A *Logistic Regression* model is trained on the TF-IDF features.
4.  *Inference:* The trained model predicts sentiment for new, unseen data (or real-time data).

## 📊 Results

The model successfully classifies sentiment across the three categories. A sample output of the sentiment distribution (from our final run):

![Sentiment Distribution Pie Chart](sentiment_output.png)
(Note: You will need to upload your saved image, named sentiment_output.png, to the repository to make this link work.)

## 📦 Setup and Installation

1.  *Clone the Repository:*
    bash
    git clone [https://github.com/tanmayfatale-code/Twitter-Sentiment-Analysis.git](https://github.com/tanmayfatale-code/Twitter-Sentiment-Analysis.git)
    
2.  *Install Dependencies* (as listed in requirements.txt):
    bash
    pip install -r requirements.txt
    
3.  *Run the Analysis:*
    bash
    python sentiment_analysis.py
    
