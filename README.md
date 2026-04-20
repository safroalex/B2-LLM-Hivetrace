# Adversarial Robustness Evaluation of GLiNER Guard Uniencoder

**Курс:** Безопасность LLM и агентных систем  
**Модель:** [`hivetrace/gliner-guard-uniencoder`](https://huggingface.co/hivetrace/gliner-guard-uniencoder) (147M params, mmBERT-based, GLiNER2 architecture)  
**Задача:** Оценка устойчивости guardrail-классификатора к adversarial text perturbations  

---

## Цель исследования

Оценить робастность мультизадачной guardrail-модели GLiNER Guard Uniencoder к различным типам adversarial текстовых пертурбаций, включая:
- символьные атаки (homoglyphs, zero-width characters, leetspeak)
- структурные трансформации (dot-splitting, markdown/HTML injection)
- семантические обёртки (academic/roleplay/negation framing)
- кросс-лингвистические атаки (transliteration RU→LAT, code-switching)

## Быстрый старт

```bash
# Создать окружение (Python 3.10+)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Запустить эксперимент
python src/run_experiment.py

# Сгенерировать графики
python src/visualize.py
```

## Структура репозитория

```
├── src/
│   ├── classifier.py       # Обёртка над GLiNER Guard Uniencoder
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
| Accuracy | 86.0% |
| TPR (Recall) | 82.0% |
| FPR | 6.0% |
| F1 | 88.7% |
| ASR | 18.0% |

### Наиболее эффективные пертурбации

| Пертурбация | ASR | Δ от baseline |
|-------------|-----|---------------|
| dot_split (b.o.m.b) | **97.0%** | +79.0 pp |
| transliteration (RU→LAT) | **94.0%** | +76.0 pp |
| code_switch (RU/EN mix) | **40.0%** | +22.0 pp |
| leetspeak (4→a, 3→e) | **39.0%** | +21.0 pp |

### Критические уязвимости

1. **Prompt injection — слепое пятно:** Baseline ASR = 70% (модель пропускает 14 из 20 prompt injection без каких-либо пертурбаций)
2. **Dot-splitting — катастрофический обход:** ASR = 97%, модель полностью теряет способность детекции при вставке точек между символами
3. **Transliteration — кросс-скриптовая уязвимость:** ASR = 94%, перевод русского текста в латиницу практически полностью обходит классификатор
4. **Hate speech & harmful instructions — высокая устойчивость:** ASR = 0% на baseline, модель стабильно детектирует явно вредоносный контент

### ASR Heatmap: Perturbation × Category

![ASR Heatmap](figures/asr_heatmap.png)

### Perturbation Effectiveness

![ASR by Perturbation](figures/asr_by_perturbation.png)

## Полный отчёт

Подробный научный отчёт с методологией, статистическим анализом и рекомендациями: **[REPORT.md](REPORT.md)**

## References

- HiveTrace GLiNER Guard Uniencoder — [HuggingFace](https://huggingface.co/hivetrace/gliner-guard-uniencoder)
- GLiNER2 Library — [GitHub](https://github.com/urchade/GLiNER)
- jhu-clsp/mmBERT — [HuggingFace](https://huggingface.co/jhu-clsp/mmBERT-small)