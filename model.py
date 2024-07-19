import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load your data
data = pd.read_csv('data.csv')

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score}')

# Save the model with the current scikit-learn version
with open("classifier.pkl", "wb") as po:
    pickle.dump(classifier, po)
