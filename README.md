# FreeTextHeuristics
Application to train a decision tree classifier using only categorical and free-text data as inputs

This application uses Kaggle's Large Purchases by the State of California (https://www.kaggle.com/datasets/sohier/large-purchases-by-the-state-of-ca) to train and test a decision tree classifier. Free-text data is difficult and tedious to manually classify. This application is an initial step in making free-text data workable for heuristic models.

This application has four modules: 
  1. A data cleaning module designed with this specific data set in mind
  2. A text parsing module, making use of NLTK packages and scikit-learn's tf-idf vectorizer
  3. A text clustering module, making use of scikit-learn's k-means clustering algorithm
  4. A decision tree module, to heuristically classify the cleansed text clusters

This application was completed as a final project for DePaul University's DSC 478: Programming Machine Learning Applications. I consider it a work in progress and intend to revisit with different data sets, clustering algorithms, and heuristics algorithms.
