import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Indian dataset load karo
df = pd.read_csv('data/IFND.csv', encoding='latin-1')

# Clean karo
df = df[['Statement', 'Label']]
df = df.dropna()

# Label convert karo — TRUE=1, Fake=0
df['label'] = df['Label'].apply(
    lambda x: 1 if x == 'TRUE' else 0)

# Features aur target
X = df['Statement'].astype(str)
y = df['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Text ko numbers mein convert karo
vectorizer = TfidfVectorizer(
    max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model train karo
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy check karo
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Model save karo
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("Indian Model saved successfully!")