import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

df = pd.read_csv('processed_reviews.csv')

# преобразование текста в векторы
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF для создания векторов
tfidf_vectorizer = TfidfVectorizer(max_features=100)
review_vectors = tfidf_vectorizer.fit_transform(df['review'].fillna('')).toarray()

# создание переменных
df['review_length'] = df['review'].apply(lambda x: len(str(x)))
scaler_length = MinMaxScaler()
review_lengths = scaler_length.fit_transform(df[['review_length']])

# тональность
np.random.seed(42)
if 'sentiment' in df.columns:
    sentiment_map = {'negative': 0.0, 'neutral': 0.5, 'positive': 1.0}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
    sentiment_scores = df[['sentiment_score']].values
else:   
    sentiment_scores = np.random.uniform(0, 1, size=(len(df), 1))

# вероятность возврата товара
return_prob = np.random.uniform(0, 1, size=(len(df), 1))

# целевые показатели в один массив
target_vectors = np.hstack([review_lengths, sentiment_scores, return_prob])

# датасет PyTorch


class VectorSeq2SeqDataset(Dataset):
    def __init__(self, input_vectors, target_vectors):
        self.input_vectors = torch.tensor(input_vectors, dtype=torch.float32)
        self.target_vectors = torch.tensor(target_vectors, dtype=torch.float32)

    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        return {
            'input': self.input_vectors[idx],
            'target': self.target_vectors[idx]
        }

train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    review_vectors, target_vectors, test_size=0.2, random_state=42
)

batch_size = 16
train_dataset = VectorSeq2SeqDataset(train_inputs, train_targets)
test_dataset = VectorSeq2SeqDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# модель Seq2Seq для числовых данных
class VectorEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(VectorEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        outputs, (hidden, cell) = self.rnn(x)
        return hidden, cell


class VectorDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(VectorDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, hidden, cell):
        decoder_input = hidden[-1].unsqueeze(1)

        outputs, (hidden, cell) = self.rnn(decoder_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions


class VectorSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(VectorSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        hidden, cell = self.encoder(x)
        output = self.decoder(hidden, cell)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = review_vectors.shape[1]
hidden_size = 128
output_size = target_vectors.shape[1]
num_layers = 1

encoder = VectorEncoder(input_size, hidden_size, num_layers).to(device)
decoder = VectorDecoder(output_size, hidden_size, num_layers).to(device)
model = VectorSeq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{num_epochs}"):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

model.eval()
test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
num_samples = 5
sample_indices = np.random.choice(len(test_inputs), num_samples, replace=False)

for i, idx in enumerate(sample_indices):
    input_vector = torch.tensor(test_inputs[idx], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_vector)

    actual = test_targets[idx]
    predicted = prediction.cpu().numpy()[0]

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'tfidf_vectorizer': tfidf_vectorizer,
    'scaler_length': scaler_length
}, 'vector_seq2seq_model.pth')