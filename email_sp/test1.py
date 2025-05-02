# Import necessary libraries
import pandas as pd  # For working with data tables
import re  # For working with regular expressions (text cleaning)
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing parts
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text into numbers
from sklearn.naive_bayes import MultinomialNB  # The machine learning model used to classify spam vs ham
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # To check how good the model is

# Step 1: Load and clean the dataset
# Load CSV file, keep only 'label' and 'email' columns
df = pd.read_csv("spam_ham_dataset_varied.csv", encoding='latin-1')[['label', 'email']]

# Rename columns to make them easier to work with
df.columns = ['label', 'message']

# Convert 'ham' to 0 and 'spam' to 1 for easier processing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 2: Define a function to clean the text messages
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove any URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation (like !, ?, ., etc.)
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces at start/end
    return text  # Return the cleaned text

# Apply the cleaning function to all messages
df['message'] = df['message'].apply(clean_text)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],  # Features (the text messages)
    df['label'],    # Labels (0 for ham, 1 for spam)
    test_size=0.8,  # Use 80% of the data for testing, 20% for training
    random_state=42  # Random seed to make results repeatable
)

# Step 4: Convert text data to numerical form using TF-IDF
# TF-IDF gives importance to words based on how often they appear
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use both single words and word pairs (bigrams)
X_train_vectors = vectorizer.fit_transform(X_train)  # Learn from training data
X_test_vectors = vectorizer.transform(X_test)  # Apply the same transformation to test data

# Step 5: Train a Naive Bayes classifier model
model = MultinomialNB()  # Create the model
model.fit(X_train_vectors, y_train)  # Train the model using the training data

# Step 6: Use the model to predict labels for the test data
y_pred = model.predict(X_test_vectors)  # Get predicted labels

# Print how well the model is performing
print("Accuracy:", accuracy_score(y_test, y_pred))  # Percentage of correct predictions
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  # Show how many were predicted right/wrong
print("Classification Report:\n", classification_report(y_test, y_pred))  # Detailed stats (precision, recall, etc.)

# Step 7: Function to check if new emails are spam or not
def predict_spam_batch(emails):
    emails_cleaned = [clean_text(email) for email in emails]  # Clean each email
    email_vectors = vectorizer.transform(emails_cleaned)  # Convert text to numbers using trained vectorizer
    predictions = model.predict(email_vectors)  # Predict spam or ham
    results = []  # Store results
    for email, pred in zip(emails, predictions):  # Loop through emails and predictions
        label = "Spam" if pred else "Ham"  # Convert 1 to "Spam" and 0 to "Ham"
        results.append((email, label))  # Save result as a tuple
    return results  # Return list of results

# Example usage: Check these example emails
emails_to_check = [
    "Hey! Are we still on for the meeting at 3pm?",  # likely ham
    "WINNER! You've been selected for a $1000 gift card. Click here!",  # spam
    "Please see attached invoice and let me know if you have any questions.",  # ham
    "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456",  # spam
    "you won rock ",  # unclear, short
    "give me back the money",  # could be aggressive tone, but not necessarily spam
    "FREE entry into our £250 weekly competition just text WIN to 80086 now!"  # spam
]

# Get the prediction results
results = predict_spam_batch(emails_to_check)

# Print the results one by one
for i, (email, prediction) in enumerate(results, start=1):
    print(f"\nEmail #{i}:\n{email}\nPrediction: {prediction}")
