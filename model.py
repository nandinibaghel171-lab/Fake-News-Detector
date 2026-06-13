import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Fake news load karo
fake_df = pd.read_csv('data/fake.csv')
fake_df['label'] = 0  # fake = 0

# Real news load karo
real_df = pd.read_csv('data/True.csv')
real_df['label'] = 1  # real = 1

# Dono combine karo
df = pd.concat([fake_df, real_df], ignore_index=True)
print("Total data:", df.shape)

# Text clean karo
df['title'] = df['title'].fillna('').astype(str)
df['text'] = df['text'].fillna('').astype(str)
df['content'] = df['title'] + ' ' + df['text']

# Features aur target
X = df['content']
y = df['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Text ko numbers mein convert karo
vectorizer = TfidfVectorizer(max_features=5000, 
                              stop_words='english')
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
print("Model saved successfully!")