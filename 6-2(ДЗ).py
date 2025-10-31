import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Завантаження моделі та даних"""
    logger.info("2.1 Завантаження попередньо навченої моделі")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    logger.info("2.2 Завантаження набору даних")
    train_dataset = load_dataset("imdb", split="train[:500]")
    eval_dataset = load_dataset("imdb", split="test[:100]")

    logger.info("2.3 Перевірка структури даних")
    logger.info(f"Розмір тренувального набору: {len(train_dataset)}")
    logger.info(f"Розмір тестового набору: {len(eval_dataset)}")

    return model, tokenizer, train_dataset, eval_dataset

def preprocess_data(dataset, tokenizer):
    """Попередня обробка даних"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Навчання моделі"""
    logger.info("4.1 Підготовка до навчання")

    # Токенізація даних
    train_tokenized = preprocess_data(train_dataset, tokenizer)
    eval_tokenized = preprocess_data(eval_dataset, tokenizer)

    # Налаштування параметрів навчання
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Ініціалізація тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    # Навчання моделі
    logger.info("4.2 Початок навчання")
    trainer.train()

    # Збереження моделі
    logger.info("4.3 Збереження моделі")
    trainer.save_model("./results")

    return trainer

def test_model(text, model, tokenizer):
    """Тестування моделі на новому тексті"""
    # Токенізація тексту
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    # Отримання передбачення
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Визначення результату
    predicted_class = "позитивний" if probs[0][1] > probs[0][0] else "негативний"
    confidence = float(max(probs[0])) * 100

    return f"Текст: '{text}'\nНастрій: {predicted_class}\nВпевненість: {confidence:.2f}%"

def main():
    # Завантаження моделі та даних
    model, tokenizer, train_dataset, eval_dataset = load_model_and_data()

    # Навчання моделі
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)

    # Тестування на прикладах
    logger.info("\nТестування навченої моделі:")
    test_texts = [
        "This movie was fantastic! Great acting and amazing plot.",
        "Terrible waste of time and money. Don't watch it.",
        "It was okay, nothing special but not bad either.",
        "I really enjoyed the special effects and music!"
    ]

    model = AutoModelForSequenceClassification.from_pretrained("./results")
    for text in test_texts:
        print("\n" + test_model(text, model, tokenizer))

if __name__ == "__main__":
    main()