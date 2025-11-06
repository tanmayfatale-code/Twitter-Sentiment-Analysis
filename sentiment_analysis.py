import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob 
# Note: TextBlob is imported but is not used for final classification 
# in the ML version; it's here for reference.

# --- 1. Data Setup (Expanded and Labeled for ML Training) ---
# NOTE: This dataset must be replaced with a much larger one (10k+ entries) 
# for accurate, real-world model training.
data_ml = {
    'Tweet': [
        "Project 3 is awesome! I learned so much about NLP and TextBlob.",
        "I'm feeling neutral about this new product launch. It's okay, I guess.",
        "This Twitter API is too complicated and frustrating. I hate it.",
        "The weather is absolutely wonderful today, such a great mood boost!",
        "Just finished the assignment; the results were neither good nor bad.",
        "The new platform update is a complete mess. Don't recommend it.",
        "The service was fast and efficient, a truly great experience.",
        "It was just an average day, nothing special happened.",
        "I am so disappointed with the quality of the new phone."
    ],
    # Manually Labeled Target Variable (0=Negative, 1=Neutral, 2=Positive)
    'Sentiment_Label': [2, 1, 0, 2, 1, 0, 2, 1, 0]
}
df = pd.DataFrame(data_ml)

# --- 2. Text Pre-processing (Cleaning) ---
def clean_tweet(text):
    # Use raw strings (r'...') to avoid SyntaxWarning
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)             # Remove the '#' symbol
    text = re.sub(r'RT[\s]+', '', text)       # Remove RT
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove the hyperlink
    text = re.sub(r'^[\s]+|[\s]+$', '', text) # Remove leading/trailing white space
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text) # Remove punctuation/special chars
    return text.lower() # Convert to lowercase for consistency

df['Cleaned_Tweet'] = df['Tweet'].apply(clean_tweet)


# --- 3. Split Data for Training and Testing ---
X = df['Cleaned_Tweet']
y = df['Sentiment_Label']
# We split the data so we can test the model on data it hasn't seen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# --- 4. Feature Engineering: TF-IDF Vectorization ---
# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data (DO NOT fit again)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# --- 5. Model Training (Logistic Regression) ---
# Initialize and train the classifier
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_tfidf, y_train)

# --- 6. Model Evaluation and Prediction ---
y_pred = model.predict(X_test_tfidf)

# Print performance metrics
print("\n" + "="*50)
print("             ML Model Performance Metrics")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report (0=Negative, 1=Neutral, 2=Positive):\n")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)'], zero_division=0))

# Convert predictions back to human-readable labels for visualization
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
y_pred_labels = pd.Series(y_pred).map(label_map)

# --- 7. Visualization ---
plt.figure(figsize=(8, 6))
plt.title('ML Model Sentiment Distribution (Test Set)')
# Use the predicted labels for the pie chart
y_pred_labels.value_counts().plot(kind='pie', 
                                  autopct='%1.1f%%', 
                                  startangle=140, 
                                  colors=['#4CAF50', '#FF9800', '#F44336'])
plt.ylabel('')
plt.show()