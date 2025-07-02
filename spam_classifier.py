import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename for clarity

# Step 2: Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix (Optional)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Predict a new message
sample_msg = ["Congratulations! You won a free iPhone. Claim now!"]
sample_vec = vectorizer.transform(sample_msg)
prediction = model.predict(sample_vec)
print("\nPrediction for sample message:", "Spam" if prediction[0] else "Ham")
