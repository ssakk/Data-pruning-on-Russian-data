import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader
from math import exp

# Путь к папке, где сохранены модель и токенизатор
model_path = 'saved_model'

# Загрузка модели и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
# Функция для вычисления перплексии
def calculate_perplexity(model, tokenizer, text, device):
    tokenize_input = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    if tokenize_input.nelement() == 0:  # Проверка на пустой токенизированный ввод
        print("Внимание: обнаружен пустой текст после токенизации.")
        return float('inf')
    tokenize_input = tokenize_input.to(device)
    with torch.no_grad():
        outputs = model(tokenize_input, labels=tokenize_input)
        loss = outputs.loss.item()
    return exp(loss)

# Подготовка датасета из локальных файлов
directory = 'C:/Users/varva/Desktop/NPlus1/texts'
files = [f for f in os.listdir(directory) if f.endswith('.txt')]
train_files = files[:5000]  # Используем для анализа и обучения
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
texts = []
for filename in train_files:
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        texts.append(text)

# Вычисление перплексии для каждого текста
perplexities = [(text, calculate_perplexity(model, tokenizer, text, device)) for text in texts]

# Фильтрация датасета
perplexities.sort(key=lambda x: x[1])
perplexity_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
a=[]
for threshold in perplexity_thresholds:
    print(f"\nОбучение модели с порогом перплексии: {threshold}")

    # Фильтрация датасета
    cutoff = int(len(perplexities) * threshold)
    filtered_dataset = [text for text, _ in perplexities[:cutoff]]

    # Подготовка DataLoader для отфильтрованного датасета
    filtered_texts = tokenizer(filtered_dataset, return_tensors='pt', max_length=512, truncation=True,
                               padding="max_length", return_attention_mask=True)
    filtered_dataset = TextDataset(filtered_texts)
    filtered_train_loader = DataLoader(filtered_dataset, batch_size=4, shuffle=True)

    # Инициализация новой модели и обучение
    new_model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2').to(device)
    train(new_model, filtered_train_loader)  # Функция обучения используется из предыдущего кода

    # Оценка новой модели на валидационном наборе данных
    val_texts = [text for text, _ in perplexities[cutoff:]]
    val_inputs = tokenizer(val_texts, return_tensors='pt', max_length=512, truncation=True, padding="max_length")
    val_dataset = TextDataset(val_inputs)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Оценка на валидационном наборе данных
    total_eval_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            outputs = new_model(input_ids, attention_mask=attention_mask, labels=labels)
            total_eval_loss += outputs.loss.item()
    avg_val_loss = total_eval_loss / len(val_loader)
    a.append(avg_val_loss)

    print(f"Средняя потеря на валидационной выборке для порога {threshold}: {avg_val_loss}")
print(a)