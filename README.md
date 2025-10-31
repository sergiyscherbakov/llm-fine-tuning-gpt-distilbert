# Fine-Tuning LLM Models: GPT-3.5 та DistilBERT

Проект демонструє два підходи до fine-tuning моделей великих мовних моделей (LLM):
1. **Fine-tuning GPT-3.5-turbo** для створення чат-бота в стилі Кузьми Скрябіна
2. **Fine-tuning DistilBERT** для аналізу сентименту кінорецензій (IMDB)

## Автор
**Розробник:** Сергій Щербаков
**Email:** sergiyscherbakov@ukr.net
**Telegram:** @s_help_2010

### 💰 Підтримати розробку
Задонатити на каву USDT (BINANCE SMART CHAIN):
**`0xDFD0A23d2FEd7c1ab8A0F9A4a1F8386832B6f95A`**

---

## 📋 Зміст
- [Опис проекту](#опис-проекту)
- [Частина 1: Fine-tuning GPT-3.5-turbo](#частина-1-fine-tuning-gpt-35-turbo)
- [Частина 2: Fine-tuning DistilBERT](#частина-2-fine-tuning-distilbert)
- [Структура проекту](#структура-проекту)
- [Технічні вимоги](#технічні-вимоги)
- [Інструкції по запуску](#інструкції-по-запуску)
- [Результати](#результати)

---

## Опис проекту

Цей проект є практичною демонстрацією fine-tuning двох різних типів моделей для різних завдань NLP (Natural Language Processing). Проект показує повний цикл роботи з моделями: від завантаження даних до навчання та тестування.

### Основні завдання:
1. **GPT-3.5-turbo**: Створити чат-бота, який імітує стиль спілкування та гумор відомого українського музиканта Кузьми Скрябіна
2. **DistilBERT**: Навчити модель розпізнавати позитивні та негативні відгуки про фільми

---

## Частина 1: Fine-tuning GPT-3.5-turbo

### Опис завдання
Навчання моделі GPT-3.5-turbo відповідати в стилі Кузьми Скрябіна, зберігаючи його унікальний гумор та манеру спілкування.

### Датасет
Датасет має бути у форматі JSONL з розмовними прикладами у стилі Кузьми Скрябіна. Кожен рядок містить структуру:
```json
{
  "messages": [
    {"role": "system", "content": "Це чат-бот, який спілкується та відповідає як Кузьма Скрябін"},
    {"role": "user", "content": "питання користувача"},
    {"role": "assistant", "content": "відповідь у стилі Кузьми"}
  ]
}
```

### Архітектура
- **Базова модель**: `gpt-3.5-turbo-0125`
- **Метод навчання**: Supervised Fine-Tuning через OpenAI API
- **Параметри**:
  - `n_epochs`: 7 (автоматично визначено)
  - `batch_size`: 1
  - `learning_rate_multiplier`: 2

### Код та методи

#### 1. Підключення до OpenAI API
```python
from google.colab import userdata
from openai import OpenAI

client = OpenAI(api_key=userdata.get('OPENAI_PERSONAL'))
```

**Параметри:**
- `api_key`: API ключ OpenAI, збережений у Colab secrets

#### 2. Завантаження файлу тренування
```python
client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)
```

**Параметри:**
- `file`: Файл у форматі JSONL з тренувальними даними
- `purpose`: "fine-tune" - вказує, що файл для fine-tuning

#### 3. Створення fine-tuning job
```python
client.fine_tuning.jobs.create(
  training_file="file-yug9ZyV0yHgcjkkZtDDJxjDi",
  model="gpt-3.5-turbo-0125"
)
```

**Параметри:**
- `training_file`: ID завантаженого файлу
- `model`: Базова модель для fine-tuning

**Результат:**
```
FineTuningJob(
  id='ftjob-XYQDqjJlsd3qoy713l8y1qJX',
  status='validating_files',
  model='gpt-3.5-turbo-0125',
  hyperparameters=Hyperparameters(
    n_epochs='auto',
    batch_size='auto',
    learning_rate_multiplier='auto'
  )
)
```

#### 4. Моніторинг процесу навчання
```python
client.fine_tuning.jobs.retrieve("ftjob-XYQDqjJlsd3qoy713l8y1qJX")
```

**Статуси:**
- `validating_files` → `running` → `succeeded`

**Фінальні параметри:**
```
hyperparameters=Hyperparameters(
  n_epochs=7,
  batch_size=1,
  learning_rate_multiplier=2
)
```

#### 5. Використання навченої моделі
```python
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::9R0iveGo",
  messages=[
    {"role": "system", "content": "Це чат-бот, який спілкується та відповідає як Кузьма Скрябін"},
    {"role": "user", "content": "розкажи якийсь анекдот"}
  ],
  temperature=0
)
```

**Параметри:**
- `model`: ID fine-tuned моделі
- `messages`: Список повідомлень (system, user)
- `temperature`: 0 для детермінованих відповідей

### Результати GPT-3.5

**Тест 1: Анекдот**
```
Запит: "розкажи якийсь анекдот"

Відповідь: "Новий анекдот: приходить мужик до лікаря і каже: 'Доктор,
у мене таке враження, що я – комп'ютер'. Лікар його тільки так дивиться.
Він каже: 'Ну, якщо чесно, мені на твої слова байдуже, але ти вийми мишку
з заду, вона мені моторошно виглядає'."
```

**Тест 2: Порівняння з базовою моделлю**
```
Запит: "Яка твоя улюблена пісня?"

Fine-tuned модель: (відповідь у стилі Кузьми)
Базова модель: "Привіт! Моя улюблена пісня - 'Мам'! А яка твоя улюблена пісня?"
```

Видно, що базова модель відповідає більш шаблонно, без характерного стилю.

---

## Частина 2: Fine-tuning DistilBERT

### Опис завдання
Навчання моделі DistilBERT для класифікації кінорецензій на позитивні та негативні.

### Датасет: IMDB Movie Reviews
- **Джерело**: HuggingFace Datasets (`imdb`)
- **Завантаження**:
  ```python
  from datasets import load_dataset

  train_dataset = load_dataset("imdb", split="train[:500]")
  eval_dataset = load_dataset("imdb", split="test[:100]")
  ```
- **Розмір**:
  - Тренувальний набір: 500 рецензій
  - Тестовий набір: 100 рецензій
- **Структура**:
  - `text`: Текст рецензії
  - `label`: 0 (негативний) або 1 (позитивний)

### Архітектура

**Базова модель**: DistilBERT (distilbert-base-uncased)
- **Тип**: Transformer-based encoder
- **Параметри**: ~66 млн
- **Особливості**: Легша версія BERT (40% менше параметрів, 60% швидше)

**Модифікації для класифікації**:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # бінарна класифікація
)
```

**Додані шари**:
- `pre_classifier`: Dense layer (768 → 768)
- `classifier`: Dense layer (768 → 2)

### Детальний опис коду

#### Клас 1: `load_model_and_data()`

**Призначення**: Завантаження моделі та датасету

**Код**:
```python
def load_model_and_data():
    """Завантаження моделі та даних"""
    logger.info("2.1 Завантаження попередньо навченої моделі")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    logger.info("2.2 Завантаження набору даних")
    train_dataset = load_dataset("imdb", split="train[:500]")
    eval_dataset = load_dataset("imdb", split="test[:100]")

    logger.info("2.3 Перевірка структури даних")
    logger.info(f"Розмір тренувального набору: {len(train_dataset)}")
    logger.info(f"Розмір тестового набору: {len(eval_dataset)}")

    return model, tokenizer, train_dataset, eval_dataset
```

**Параметри**:
- `model_name`: "distilbert-base-uncased" - ім'я моделі на HuggingFace
- `num_labels`: 2 - кількість класів (позитивний/негативний)

**Повертає**:
- `model`: Модель DistilBERT з класифікаційною головою
- `tokenizer`: Токенізатор для обробки тексту
- `train_dataset`, `eval_dataset`: Датасети для навчання та оцінки

#### Клас 2: `preprocess_data(dataset, tokenizer)`

**Призначення**: Токенізація та підготовка даних

**Код**:
```python
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
```

**Параметри токенізації**:
- `padding="max_length"`: Доповнює всі послідовності до однакової довжини
- `truncation=True`: Обрізає тексти, що перевищують max_length
- `max_length=128`: Максимальна довжина токенів (баланс між якістю та швидкістю)
- `batched=True`: Обробка пакетами для швидкості

**Процес**:
1. Текст розбивається на токени (підслова)
2. Токени конвертуються в ID
3. Додаються спеціальні токени [CLS], [SEP]
4. Створюється attention mask

#### Клас 3: `train_model(model, tokenizer, train_dataset, eval_dataset)`

**Призначення**: Навчання моделі

**Код**:
```python
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
```

**Параметри TrainingArguments**:
- `output_dir="./results"`: Папка для збереження результатів
- `num_train_epochs=3`: Кількість епох навчання
- `per_device_train_batch_size=16`: Розмір батчу для тренування
- `per_device_eval_batch_size=16`: Розмір батчу для оцінки
- `warmup_steps=50`: Кількість кроків прогріву для learning rate
- `weight_decay=0.01`: L2 регуляризація для запобігання перенавчанню
- `logging_steps=10`: Частота логування
- `evaluation_strategy="epoch"`: Оцінка після кожної епохи
- `save_strategy="epoch"`: Збереження після кожної епохи
- `load_best_model_at_end=True`: Завантажити найкращу модель після навчання

**Trainer**:
- Автоматично керує циклом навчання
- Обчислює loss та градієнти
- Виконує backpropagation
- Оновлює ваги моделі

#### Клас 4: `test_model(text, model, tokenizer)`

**Призначення**: Тестування моделі на новому тексті

**Код**:
```python
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
```

**Параметри**:
- `return_tensors="pt"`: Повертає PyTorch тензори
- `truncation=True`: Обрізає довгі тексти
- `max_length=128`: Максимальна довжина

**Процес інференсу**:
1. Токенізація вхідного тексту
2. Прогін через модель → отримання logits
3. Застосування softmax для отримання ймовірностей
4. Вибір класу з найвищою ймовірністю

#### Функція `main()`

**Повний пайплайн**:
```python
def main():
    # 1. Завантаження моделі та даних
    model, tokenizer, train_dataset, eval_dataset = load_model_and_data()

    # 2. Навчання моделі
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)

    # 3. Тестування на прикладах
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
```

### Результати DistilBERT

#### Консольний вивід під час навчання

```
INFO:__main__:2.1 Завантаження попередньо навченої моделі
Some weights of DistilBertForSequenceClassification were not initialized
from the model checkpoint at distilbert-base-uncased and are newly initialized:
['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it
for predictions and inference.

INFO:__main__:2.2 Завантаження набору даних
INFO:__main__:2.3 Перевірка структури даних
INFO:__main__:Розмір тренувального набору: 500
INFO:__main__:Розмір тестового набору: 100

INFO:__main__:4.1 Підготовка до навчання
Map: 100%|██████████| 500/500 [00:00<00:00, 2547.83 examples/s]
Map: 100%|██████████| 100/100 [00:00<00:00, 2834.12 examples/s]

INFO:__main__:4.2 Початок навчання
```

#### Метрики навчання

**Epoch 1:**
```
Step 10: Loss = 0.6234
Step 20: Loss = 0.4521
Step 30: Loss = 0.3142
Validation Loss = 0.018526
```

**Epoch 2:**
```
Step 10: Loss = 0.0234
Step 20: Loss = 0.0156
Step 30: Loss = 0.0089
Validation Loss = 0.000835
```

**Epoch 3:**
```
Step 10: Loss = 0.0034
Step 20: Loss = 0.0021
Step 30: Loss = 0.0012
Validation Loss = 0.000569
```

**Графік навчання:**
- Training Loss: 0.114800 → 0.001600 → 0.000700
- Validation Loss: 0.018526 → 0.000835 → 0.000569

#### Проблема: Модель класифікує все як негативне

**Тести після навчання:**

```
Текст: 'This movie was fantastic! Great acting and amazing plot.'
Настрій: негативний
Впевненість: 99.92%

Текст: 'Terrible waste of time and money. Don't watch it.'
Настрій: негативний
Впевненість: 99.92%

Текст: 'It was okay, nothing special but not bad either.'
Настрій: негативний
Впевненість: 99.93%

Текст: 'I really enjoyed the special effects and music!'
Настрій: негативний
Впевненість: 99.90%
```

**Додаткові тести:**

```
Текст: 'I think this movie has amazing special effects and great story!'
Настрій: негативний
Впевненість: 99.92%

Текст: 'The acting was terrible and the plot made no sense'
Настрій: негативний
Впевненість: 99.91%

Текст: 'This is the best movie I have ever seen!'
Настрій: негативний
Впевненість: 99.90%

Текст: 'The movie was just ok, nothing special'
Настрій: негативний
Впевненість: 99.92%
```

#### Аналіз проблеми

**Можливі причини:**
1. **Дисбаланс класів**: Можливо, в тренувальних даних переважають негативні відгуки
2. **Малий розмір датасету**: 500 прикладів замало для повноцінного навчання
3. **Переобучення**: Loss зменшується занадто швидко (0.114 → 0.0007)
4. **Проблема з токенізацією**: Можлива невідповідність між тренуванням та інференсом
5. **Ініціалізація класифікатора**: Новий шар класифікатора може бути неправильно ініціалізований

**Рішення**:
- Збільшити датасет до 5000+ прикладів
- Використати balanced sampling
- Зменшити learning rate
- Додати early stopping
- Перевірити розподіл класів у датасеті

---

## Структура проекту

```
6/
├── lec6.ipynb                    # Jupyter notebook з Fine-tuning GPT-3.5
├── r_d_lesson_6_2.ipynb          # Jupyter notebook з Fine-tuning DistilBERT
├── 6-2(ДЗ).py                    # Python скрипт DistilBERT (standalone версія)
├── 6-2 клауд гугл.txt            # Текстовий файл з кодом
├── hz-6.docx                     # Документація завдання
├── lesson-6.pdf                  # PDF з лекційним матеріалом
├── README.md                     # Ця документація
└── .gitignore                    # Git ignore файл
```

---

## Технічні вимоги

### Python бібліотеки

```bash
# Для GPT-3.5
openai>=1.0.0

# Для DistilBERT
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0

# Для візуалізації
matplotlib>=3.7.0

# Для логування
logging (стандартна бібліотека)
```

### Системні вимоги

- **Python**: 3.8+
- **GPU**: Рекомендується для DistilBERT (CUDA-compatible)
- **RAM**: Мінімум 8GB
- **Disk Space**: 2GB для моделей та даних

---

## Інструкції по запуску

### 1. Fine-tuning GPT-3.5-turbo

**Крок 1: Підготовка даних**
```bash
# Створіть файл mydata.jsonl з розмовними прикладами
# Формат: {"messages": [{"role": "system", "content": "..."}, ...]}
```

**Крок 2: Налаштування API ключа**
```python
# У Google Colab
from google.colab import userdata
api_key = userdata.get('OPENAI_PERSONAL')

# Локально
export OPENAI_API_KEY='your-api-key-here'
```

**Крок 3: Запуск навчання**
```python
# Відкрийте lec6.ipynb у Google Colab
# Виконайте всі клітинки послідовно
```

**Крок 4: Використання моделі**
```python
from openai import OpenAI

client = OpenAI(api_key='your-key')
completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal::YOUR_MODEL_ID",
    messages=[
        {"role": "system", "content": "Це чат-бот, який спілкується та відповідає як Кузьма Скрябін"},
        {"role": "user", "content": "Ваше питання"}
    ]
)
print(completion.choices[0].message.content)
```

### 2. Fine-tuning DistilBERT

**Крок 1: Встановлення залежностей**
```bash
pip install torch transformers datasets numpy pandas matplotlib
```

**Крок 2: Запуск Python скрипта**
```bash
python "6-2(ДЗ).py"
```

**Крок 3: Або використання Jupyter Notebook**
```bash
# Відкрийте r_d_lesson_6_2.ipynb
jupyter notebook r_d_lesson_6_2.ipynb
# Виконайте всі клітинки
```

**Крок 4: Тестування моделі**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Завантаження моделі
model = AutoModelForSequenceClassification.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Тест
text = "This movie is amazing!"
result = test_model(text, model, tokenizer)
print(result)
```

---

## Висновки та навчальні цілі

### Що було реалізовано:

1. **Fine-tuning GPT-3.5**:
   - ✅ Успішне налаштування моделі для генерації тексту у специфічному стилі
   - ✅ Використання OpenAI API для керування процесом навчання
   - ✅ Практика роботи з форматом JSONL для розмовних даних

2. **Fine-tuning DistilBERT**:
   - ✅ Завантаження та підготовка датасету IMDB
   - ✅ Налаштування Transformer моделі для класифікації
   - ✅ Використання HuggingFace Trainer API
   - ⚠️ Виявлено проблему з класифікацією (потребує додаткового налаштування)

### Навчальні моменти:

1. **Важливість якості даних**: Розмір та баланс датасету критично впливають на результат
2. **Різні підходи до fine-tuning**: API-based (OpenAI) vs custom training (HuggingFace)
3. **Моніторинг метрик**: Важливість відстеження loss та валідаційних показників
4. **Debugging моделей**: Навички діагностики проблем з навченими моделями

---

## Ліцензія

Цей проект створено в навчальних цілях. Використовуйте на власний розсуд.

---

## Контакти

Для питань та пропозицій:
- **Email**: sergiyscherbakov@ukr.net
- **Telegram**: @s_help_2010

### 💰 Підтримати розробку
Задонатити на каву USDT (BINANCE SMART CHAIN):
**`0xDFD0A23d2FEd7c1ab8A0F9A4a1F8386832B6f95A`**
