import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import time

def analyze_review_distribution(data):
    sentiment_counts = data['label'].value_counts()
    sentiment_labels = ['Positive', 'Negative']
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_labels, y=sentiment_counts, palette="viridis")
    plt.title('Distribution of Positive vs. Negative Reviews')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Call the function to plot the distribution


# Function to load data
def load_data(file_path):
    reviews = []
    with open(file_path, 'r') as file:
        review = {}
        for line in file:
            line = line.strip()
            if line.startswith("product/productId"):
                if review:
                    reviews.append(review)
                    review = {}
                review["productId"] = line.split(": ")[1].strip()
            if line.startswith("review/score"):
                review["score"] = float(line.split(": ")[1])
            if line.startswith("review/text"):
                review["text"] = line.split("review/text: ")[1]
        if review:
            reviews.append(review)
    data = pd.DataFrame(reviews)
    data['label'] = data['score'].apply(lambda x: 1 if x >= 4 else 0)
    return data

def predict_by_product_id(data, model, vectorizer, product_id):
    # Debug: Print the dataset for matching
    print(data[data['productId'] == product_id])  # Debugging step
    review = data[data['productId'] == product_id]
    if review.empty:
        return f"No review found for productId: {product_id}"
    text = review['text'].iloc[0]
    features = vectorizer.transform([preprocess_text(text)])
    prediction = model.predict(features)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return f"Review: {text}\nPrediction: {sentiment}"


# Preprocessing function
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

# Train the model
def train_model(data):
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, vectorizer

# Main script
if __name__ == "__main__":
    file_path = 'finefoods.txt'
    data = load_data(file_path)
    print("Data Loaded Successfully!")
    
    # Train the model
    model, vectorizer = train_model(data)
    
    # Predict for a specific productId
    product_id = "B006K2ZZ7K"  # Replace with your productId
    result = predict_by_product_id(data, model, vectorizer, product_id)
    print(result)
    analyze_review_distribution(data)
    timings = {
        'Data Loading': 2.5,  # Example time in seconds
        'Model Training': 15.3,
        'Prediction Execution': 0.8
    }
    timing_table = pd.DataFrame(list(timings.items()), columns=["Step", "Time (seconds)"])

    # Plot the timings as a bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Step", y="Time (seconds)", data=timing_table, palette="coolwarm")
    plt.title("Execution Time for Different Steps")
    plt.xlabel("Step")
    plt.ylabel("Time (seconds)")
    plt.show()
