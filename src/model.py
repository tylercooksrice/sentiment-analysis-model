import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Preprocessing function
def preprocess_text_simple(text):
    return text.lower()  # Only lowercase, no punctuation removal

# Train a less accurate model with Bag of Words
def train_less_accurate_model(data):
    # Use simpler preprocessing
    data['cleaned_text'] = data['text'].apply(preprocess_text_simple)
    
    # Use Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=100)  # Only 100 features
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['label']
    
    # Use a smaller training dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)  # 80% for testing
    
    # Train the logistic regression model
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy (Less Accurate Model with Bag of Words):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, vectorizer

# Predict for a specific productId
def predict_by_product_id(data, model, vectorizer, product_id):
    review = data[data['productId'] == product_id]
    if review.empty:
        return f"No review found for productId: {product_id}"
    text = review['text'].iloc[0]
    features = vectorizer.transform([preprocess_text_simple(text)])
    prediction = model.predict(features)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return f"Review: {text}\nPrediction: {sentiment}"

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

# Main script
if __name__ == "__main__":
    file_path = 'finefoods.txt'  # Replace with your dataset path
    data = load_data(file_path)
    print("Data Loaded Successfully!")
    
    # Train the less accurate model with Bag of Words
    less_accurate_model, less_accurate_vectorizer = train_less_accurate_model(data)
    
    # Predict for a specific productId
    product_id = "B006K2ZZ7K"  # Replace with your productId
    result_less_accurate = predict_by_product_id(data, less_accurate_model, less_accurate_vectorizer, product_id)
    print("Less Accurate Model Prediction:\n", result_less_accurate)
