# Adversarial Robustness Evaluation of HiveTrace Guard

**Курс:** Безопасность LLM и агентных систем  
**Модель:** `hivetrace/hivetrace-guard-base-2025-10-23` (ModernBERT, 22 layers, 768 hidden, binary classifier)  
**Задача:** Оценка устойчивости guardrail-классификатора к adversarial text perturbations  

---

## Цель исследования

Оценить робастность guardrail-модели HiveTrace Guard к различным типам adversarial текстовых пертурбаций, включая:
- символьные атаки (homoglyphs, zero-width characters, leetspeak)
- структурные трансформации (dot-splitting, markdown/HTML injection)
- семантические обёртки (academic/roleplay/negation framing)
- кросс-лингвистические атаки (transliteration RU→LAT, code-switching)

## Быстрый старт

```bash
# Создать окружение (Python 3.10+)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Установить HF_TOKEN (модель приватная)
export HF_TOKEN="your_token_here"

# Запустить эксперимент
python src/run_experiment.py

# Сгенерировать графики
python src/visualize.py
```

## Структура репозитория

```
├── src/
│   ├── classifier.py       # Обёртка над HiveTrace Guard (transformers pipeline)
│   ├── dataset.py           # Курированный датасет (150 промптов, 7 категорий)
│   ├── perturbations.py     # 17 функций adversarial-пертурбаций
│   ├── run_experiment.py    # Основной скрипт эксперимента
│   └── visualize.py         # Генерация графиков
├── results/
│   ├── raw_results.csv      # Сырые результаты (2700+ записей)
│   └── analysis.json        # Агрегированные метрики
├── figures/                  # Визуализации
├── REPORT.md                # Полный научный отчёт
└── requirements.txt
```

## Ключевые результаты

### Baseline (без пертурбаций)

| Метрика | Значение |
|---------|----------|
| Accuracy | **96.7%** |
| TPR (Recall) | **96.0%** |
| FPR | 2.0% |
| F1 | **97.5%** |
| ASR | **4.0%** |

### Наиболее эффективные пертурбации

| Пертурбация | ASR | Δ от baseline |
|-------------|-----|---------------|
| dot_split (b.o.m.b) | **60.0%** | +56.0 pp |
| transliteration (RU→LAT) | **36.0%** | +32.0 pp |
| code_switch (RU/EN mix) | **20.0%** | +16.0 pp |
| char_swap | **17.0%** | +13.0 pp |

### Критические уязвимости

1. **Dot-splitting — главная уязвимость:** ASR = 60%, модель теряет значительную часть способности детекции при вставке точек между символами (особенно для harmful instructions: 65%, hate speech: 70%, social engineering: 70%)
2. **Transliteration — кросс-скриптовая уязвимость:** ASR = 36%, перевод русского текста в латиницу существенно обходит классификатор (harmful instructions: 60%)
3. **Code-switching:** ASR = 20%, смешение RU/EN снижает эффективность детекции
4. **Высокая устойчивость к обёрткам:** markdown_wrap, html_comment, base64_hint — ASR = 0%, модель полностью устойчива к структурным обёрткам (хотя html_comment вызывает рост FPR)

### ASR Heatmap: Perturbation × Category

![ASR Heatmap](figures/asr_heatmap.png)

### Perturbation Effectiveness

![ASR by Perturbation](figures/asr_by_perturbation.png)

## Полный отчёт

Подробный научный отчёт с методологией, статистическим анализом и рекомендациями: **[REPORT.md](REPORT.md)**

## References

- HiveTrace Guard — приватная модель на HuggingFace
- ModernBERT Architecture — [HuggingFace](https://huggingface.co/docs/transformers/model_doc/modernbert)
- jhu-clsp/mmBERT-base — [HuggingFace](https://huggingface.co/jhu-clsp/mmBERT-base)