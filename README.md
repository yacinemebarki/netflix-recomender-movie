# 🎬 Netflix Recommender for New Users

## 📖 Introduction

When we open a new account on Netflix, the platform usually recommends random movies that may not fit our **age**, **culture**, or **preferences**.  
This project solves that problem by building a **machine learning–based recommender system**.  
Our model predicts which movies are most suitable for a new user and sorts them in **descending order of relevance**, giving more personalized and meaningful recommendations.

## 🚨 Problem Statement
The main challenge addressed here is the **cold start problem** in recommender systems:  
> *How can we recommend relevant movies to a new user without having their past viewing history?*

## Proposed Methods
### 🔹 Data Preprocessing
- Removing duplicates and null values  
- Creating histograms of numerical features  
- Encoding categorical data
### 🔹 Three-Step Prediction Pipeline
1. **recommendation**: We use past user reviews of movies along with unsupervised clustering to divide movies into three categories based on their similarity to the user’s profile: Super Recommend, Recommend, or Not Recommend.  
2. **people category**:We use age, nationality, subscription plan, and gender to classify users into three categories. (In this project, we only consider users from USA and Canada.)
3. **model**:We use the user category (age, nationality, subscription plan, gender) together with movie features to determine whether a movie should be classified as Super Recommend, Recommend, or Not Recommend.
### 🔹 Bot
This bot receives the user’s information, assigns them to the correct group, and then searches the movie dataset to sort and recommend the best movies for that user using our model’s probability calculations.

## Methodology

### 📊 Data Exploration and Preprocessing
- **Data Cleaning**: Removed duplicate rows and null values.  
- **Feature Engineering**: Created histograms of numerical features, encoded categorical data, and performed exploratory data analysis.  
- **Visualization**: Utilized heatmaps, box plots, and histograms to understand data distributions and correlations.  

### 🔻 Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduced movie feature dimensions while retaining maximum information.  
- **K-Means Clustering (Unsupervised)**: Grouped movies into three recommendation clusters (**Super Recommend, Recommend, Not Recommend**) with a silhouette score of **0.79**.  

### 🤖 Model Training and Evaluation
- **User Categorization Model**:  
  - Classified users into groups based on **age, nationality, subscription plan, and gender**.  
  - Implemented using **K-Means clustering** with an accuracy of **0.50**.  

- **Recommendation Model**:  
  - Implemented with a **Gradient Boosting Classifier** inside a Scikit-Learn Pipeline:  
    ```python
    modelG = Pipeline([
        ("preprocess", pre),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    ```
  - Achieved an accuracy of **0.60** in predicting whether a movie should be **Super Recommend, Recommend, or Not Recommend**.  

### 📈 Performance Metrics
- **Movie Recommendation (k-Means**: Silhouette Score = **0.79**  
- **User Categorization (K-Means)**: Silhouette Score = **0.50**  
- **Recommendation Model (Gradient Boosting)**: Accuracy = **0.60**
## 🔮 Future Work
- Incorporate **collaborative filtering** to improve recommendations by leveraging similarities between users.  
- Expand dataset to include more countries, languages, and diverse demographics.  
- Enhance the model with **deep learning techniques** (e.g., Neural Collaborative Filtering, Transformers).  
- Add real-time recommendation updates as user behavior evolves.  
- Deploy the system as a **web app with an interactive UI** for better accessibility.

## ✅ Conclusion
This project tackles the **cold start problem** in recommendations by classifying users into demographic groups and predicting movie relevance with clustering and Gradient Boosting.  
The system provides more personalized recommendations for new users compared to random suggestions and lays the groundwork for future improvements.
  





