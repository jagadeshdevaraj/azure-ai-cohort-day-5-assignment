import os
os.system('pip install nltk scikit-learn matplotlib seaborn torch torchvision')

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

print("Current working directory:", os.getcwd())
# Load the dataset
#df = pd.read_csv('./IMDB_Dataset.csv')  # Update with the correct path to your dataset
# Ensure the correct path to your dataset
 
dataset_path = './Users/sajaykumr/IMDB_Dataset.csv'
# List contents of the current directory
directory_contents = os.listdir('.')
print("Contents of the current directory:")
for item in directory_contents:
    print(item)

df = pd.read_csv(dataset_path)
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
else:
    print(f"File not found at {dataset_path}. Please check the file path.")

# Inspect the dataset
print(df.head())
print(df.info())

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'\W', ' ', text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and apply stemming/lemmatization
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the dataset
df['processed_review'] = df['review'].apply(preprocess_text)
print(df.head())

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform the processed reviews
X = tfidf.fit_transform(df['processed_review']).toarray()

# Encode target variable
y = pd.get_dummies(df['sentiment'], drop_first=True).values.ravel()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SentimentClassifier model
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
model = SentimentClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Training loop
num_epochs = 10
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Plotting Loss Curve
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

# Predict on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).round().numpy()

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy:.4f}')

# Display sample predictions
sample_indices = np.random.choice(len(y_test), 10, replace=False)
for i in sample_indices:
    print(f'Review: {df.iloc[i]["review"][:100]}...')
    print(f'Actual Sentiment: {y_test[i]}, Predicted Sentiment: {predictions[i][0]}')
