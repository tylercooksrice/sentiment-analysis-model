# predict-ratings-model

Predicting model

This project performs two key analyses on Amazon Fine Foods Reviews: predicting the helpfulness of reviews and conducting sentiment analysis. The dataset, spanning from 1999 to 2012, contains over 500,000 reviews with features like product ID, user ID, helpfulness score, and review text. For helpfulness prediction, we used a Random Forest Regressor to model the relationship between review length, TF-IDF features, and helpfulness ratios, achieving strong results. For sentiment analysis, we used Logistic Regression with TF-IDF to classify reviews as positive or negative based on the review text, achieving an accuracy of 87%. The project demonstrates effective use of machine learning models like Random Forest and Logistic Regression for text-based tasks, with optimizations to handle large datasets and feature extraction.

[dataset used](https://snap.stanford.edu/data/web-FineFoods.html)
