import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
tokenizer.pad_token = tokenizer.eos_token

directory = 'C:/Users/varva/Desktop/NPlus1/texts'

# Мы будем собирать тексты в один большой список
train_texts = []
val_texts = []

# Читаем txt файлы и разделяем их на обучающую и валидационную выборки
files = [f for f in os.listdir(directory) if f.endswith('.txt')]
train_files = files[:5000]  # 5000 файлов для обучения
val_files = files[5000:7000]  # 2000 файлов для валидации

for filename in train_files:
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        train_texts.append(text)

for filename in val_files:
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        val_texts.append(text)

# Токенизация
train_inputs = tokenizer(train_texts, return_tensors='pt', max_length=512, truncation=True, padding="max_length")
val_inputs = tokenizer(val_texts, return_tensors='pt', max_length=512, truncation=True, padding="max_length")

# Подготовка DataLoader для обучающего и валидационного наборов
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

train_dataset = TextDataset(train_inputs)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = TextDataset(val_inputs)
val_loader = DataLoader(val_dataset, batch_size=4)

def train(model, train_loader):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for epoch in range(1):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

train(model, train_loader)

model.eval()

# Оценка модели на валидационном наборе данных
total_eval_loss = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_eval_loss += outputs.loss.item()

avg_val_loss = total_eval_loss / len(val_loader)
print(f"Средняя потеря на валидационной выборке: {avg_val_loss}")

model_save_path = 'saved_model'
tokenizer_save_path = 'saved_model'

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f"Модель сохранена в {model_save_path}")
print(f"Токенизатор сохранен в {tokenizer_save_path}")
