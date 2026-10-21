# Project review and completion plan — 2026-10-17

Обзор текущего кода, историй обучения и сохранённых результатов. Новые модели не
обучались; полный evaluation не запускался. Это исследовательская оценка, а не
утверждение, что каждая строка репозитория прошла аудит корректности.

## Основной вывод

Прямое улучшение item-level рекомендаций для mid/tail/unseen за счёт SID пока не
продемонстрировано. Однако новые prefix-эксперименты показывают полезный сигнал
выше двух popularity controls. Необходимо различать способность выбрать область
каталога, способность ранжировать товары в ней и причинный вклад семантики SID.
Третье пока не изолировано сравнением с несемантическими кодами той же структуры.

Разумная конечная цель: воспроизводимое исследование возможностей и ограничений
semantic-ID retrieval в разреженном marketplace-каталоге, с одной проверенной
гибридной альтернативой. Положительный результат не является условием завершения.

## Что уже сделано

- Основной Amazon pipeline, BGE embeddings, RQ-VAE, несколько SID/tokenizer
  вариантов, GenRec A/B/C/D, constrained beam search и групповые метрики.
- SASRec, Content Two-Tower, ItemKNN, GRU4Rec, LightGCN, adapted TIGER.
- Проверки PAD/masking, порядка split, SID/catalog/checkpoint соответствия;
  основной корпус тестов находится в `tests/`.
- Диагностики prefix survival, teacher-forced likelihood, зависимости от train
  priors, shuffled histories, prior correction и trie normalization.
- Четыре RQ-кода с уникальным matching внутри bucket, final/all-depth behavioral
  supervision. По сохранённым результатам это не решило rare-item ranking.
- Prefix expansion, size normalization, quota hybrids и popularity controls.
- Main, Industrial и MovieLens имеют полные сохранённые оценки. Beauty пока
  представлен smoke-результатом. Эти датасеты не являются взаимозаменяемыми.

Исторические таблицы: [results](../results/overview.md), [research log](../research_log.md),
[baseline development](../results/baseline_development.md), [archive](../archive.md).

## Новые результаты: validation, 10 000 пользователей

Источники: [BGE reranking](../../../artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid10000_v2.json) и
[Two-Tower-vector reranking](../../../artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid10000_v2_tt.json).
Проверены совпадения checkpoint SHA, dataset/SID hashes, ordered users, beam,
выборки и результатов controls. Это парное сравнение reranking embeddings.
Один checkpoint, один seed пользователей и один random-control seed.

Все значения ниже — проценты. Depth 2 candidate coverage:

| Group | n | GenRec | Top train prefixes | Matched frequency + size |
|---|---:|---:|---:|---:|
| all | 10000 | 21.62 | 12.03 | 9.08 |
| head | 3003 | 41.76 | 31.07 | 23.08 |
| mid | 2402 | 16.69 | 5.83 | 4.25 |
| tail | 4040 | 11.34 | 2.95 | 2.65 |
| unseen | 555 | 8.83 | 1.98 | 1.08 |

Средний pool: GenRec 2936 items; matched control 2916; top-popularity 4548.
Таким образом, преимущество над matched control не объясняется просто большим
средним пулом. Matching приближённый, по factor-two bins, а histogram bins
заимствован у предсказания модели. Это не полностью независимый baseline и не
доказательство причинного вклада семантики.

Для tail модель покрывает 458 targets, matched control — 107: только модель
покрывает 383, только контроль — 32. Для unseen: 49 против 6, парные различия
46 против 3. Это более убедительное наблюдение, чем единичный cold-start hit.
На depth 3 coverage снижается до 1.44% tail и 0.72% unseen.

Recall@50 и абсолютные hits:

| Method | All R@50 | Head hits | Mid hits | Tail hits | Unseen hits |
|---|---:|---:|---:|---:|---:|
| Full-SID GenRec beam | 6.56 | 636 | 14 | 5 | 1 |
| Prefix 2 + BGE mean-profile | 2.98 | 243 | 23 | 29 | 3 |
| Prefix 2 normalized + BGE | 4.22 | 381 | 26 | 13 | 2 |
| Hybrid depth 2 + BGE | 6.34 | 605 | 16 | 11 | 2 |
| Hybrid depth 2 + TT vectors | 6.39 | 609 | 18 | 10 | 2 |
| Hybrid depth 3 + TT vectors | 6.46 | 619 | 16 | 10 | 1 |

Hybrid сохраняет top-40 полного SID beam и заменяет до 10 последних позиций.
Неизменные top-10/20 здесь гарантированы конструкцией и не являются открытием.
Нормализация улучшает aggregate относительно raw expansion, но может уменьшать
tail hits. Полезного улучшения без компромиссов пока нет. Один–три unseen hits
недостаточны для устойчивого вывода о превосходстве ранжирования.

`exact` в JSON означает полный SID beam, не exhaustive item scoring.

## Важный уже существующий baseline

[Content Two-Tower all-catalog](../../../artifacts/main/evaluation/two_tower_metrics.json) имеет:

| Model | All R@50 | Mid R@50 | Tail R@50 | Unseen R@50 |
|---|---:|---:|---:|---:|
| Content Two-Tower | 4.476 | 1.526 | 0.785 | 0.652 |
| GenRec C, Two-Tower SID | 5.669 | 0.554 | 0.065 | 0.048 |

Это исторические запуски с неодинаковым selection rule и неполной одинаковой
provenance; таблица мотивирует повторную согласованную оценку, а не заменяет её.
Тем не менее, content-based модель уже давала 41 unseen hit@50 против трёх у
данного GenRec. Не следует ограничивать сравнение cold-start моделями с ID lookup.

Нынешний prefix reranker усредняет BGE/TT **item vectors** истории. Он не использует
обученную user tower `ContentTwoTower.encode`. Следующий дешёвый эксперимент —
ранжировать те же кандидаты обученным score `encode(history) @ item_vectors.T`.
Нужны одновременно full-catalog TT и prefix-candidates + TT; иначе улучшение
может целиком происходить от ranker, без полезного вклада генеративного retrieval.

## Что действительно утверждает литература

- [TIGER, §4.3](https://arxiv.org/html/2305.05065v3): cold-start эксперимент
  удаляет 5% test items из train, добавляет unseen-кандидатов по совпадению первых
  трёх semantic tokens и регулирует их максимальную долю через epsilon. Это
  дополнительная retrieval policy, а не только обычный full-SID beam.
- [LIGER, §2.3–2.4](https://arxiv.org/html/2411.18814v2): авторы наблюдают слабый
  cold-start у TIGER и сравнивают с dense retrieval при сопоставимой информации.
  Их hybrid добавляет cold-item set к generative candidates перед dense ranking;
  его успех нельзя автоматически приписывать одному SID decoding.
- [LETTER](https://arxiv.org/html/2405.07314v2): рассматривает code-assignment
  bias; использует collaborative/diversity regularization и ranking-guided
  objective. Это более сильное вмешательство, чем замена ID на RQ-коды.
- [Can Generative Recommendation Reach Cold Items?](https://arxiv.org/abs/2607.21101),
  препринт июля 2026: исследует temporal cold start, поддержку токенов/префиксов
  обучением и ограничения генерации новых путей. Тематика близка нашим
  наблюдениям; не следует объявлять сам факт cold-start ограничения новым.

Положительные публикации существуют. Их постановки, baseline information,
candidate policies и cold-item определения необходимо сравнивать явно.
Semantic IDs дают механизм sharing, но не гарантию полезного item-level ranking.

## Незакрытые методологические вопросы

1. **Нет изолированного SID-vs-random контроля.** Переставить целые SID между
   товарами, сохранив trie, vocab и длину, затем заново обучить тот же GenRec.
   Для ограничения popularity confounding переставлять внутри заранее заданных
   train-frequency strata. Инициализацию и weight tying держать одинаковыми;
   интерпретировать контроль как разрушение item-to-semantic-address связи,
   а не как отдельное измерение роли RQ initialization.
2. **Cold-start ограничен постановкой.** Текущие unseen имеют ноль train
   interactions, но их metadata доступны tokenizer и trie. Per-user leave-two-out
   не является global temporal split. Dynamic insertion и стабильность SID при
   изменении каталога не проверены. Matching c4 может переадресовывать старые items.
3. **Selection rules различаются.** GenRec выбирается по validation token loss,
   новые baseline — по all NDCG@10, Two-Tower — warm R@20. Особенно для редких
   targets это разные цели. Не выбирать лучший метод по уже просмотренному test.
4. **TIGER не закончен.** На момент обзора history содержит два epochs с ростом
   validation NDCG; это adapted baseline, не отрицательная репликация paper.
5. **Сильная разреженность и короткие истории.** По сохранённому data audit медиана
   полной истории — 7, медиана train interactions/item — 2. Смешанные категории
   и фильтрация оставляют только часть реальной последовательности событий.
   Семантическая близость не равна next-item совместимости: substitutes/complements
   могут вести себя по-разному.
6. **Не все train-cold определения одинаковы.** Первый item истории встречается
   в train, но не бывает target этой последовательности. Для диагностики нужно
   различать train interactions, target support, position-specific token support
   и joint-prefix support. Отсутствие support не делает токен математически
   невозможным, но означает отсутствие соответствующих положительных примеров.
7. **Текстовый encoder не проверен контролируемо.** Сейчас BGE-small: 384 — это
   размерность embedding, max sequence length — 512 токенов; текст до encoder
   также обрезается до 4000 символов. Больший max length не восстановит уже
   удалённый текст. Перед новым encoder нужен аудит truncation и качества текста.
8. **Нет финальной статистической оценки.** Нужны per-user ranks/hits, paired
   bootstrap CI и несколько training/control seeds для финальных сравнений.
   Нельзя трактовать несколько дополнительных unseen hits как устойчивый эффект.

## Ограниченный план завершения

### 1. Заморозить постановку и собрать честную таблицу

Один main snapshot, единые users/candidates/seen filtering, train-only groups,
одна validation selection policy для новых сопоставимых запусков. Завершить TIGER
до заранее заданного budget/early stopping. Переоценить существующие SASRec,
Two-Tower, GenRec и новые baselines на одном протоколе. Popularity уже реализован
в `src/models/baseline_popularity.py`, но пропускает unseen и не сохраняет новый
формат отчёта; нужна адаптация, не новый алгоритм.

Оставить исторические результаты отдельно: повторная evaluation не меняет того,
как ранее выбирались checkpoints. Старый test уже многократно просмотрен; честно
назвать его exploratory. Независимая подтверждающая оценка требует нового,
заранее определённого holdout либо отдельного датасета.

### 2. Закрыть два объяснения до нового обучения

- Повторить matched controls с несколькими control seeds на фиксированных users.
  Сравнить реальные и shuffled histories при том же retrieval budget; это
  проверка полезности истории, а не доказательство роли семантического tokenizer.
- Использовать замороженную обученную Two-Tower user tower для prefix-2/3
  candidate reranking; сравнить с её full-catalog retrieval и текущим mean-profile.
  Разложить coverage и Recall@50 given covered. Измерять реальные latency/memory,
  а не объявлять малый pool автоматическим выигрышем скорости.

### 3. Один причинный контроль и один финальный treatment

Semantic vs permuted SID — ключевой контроль исходной гипотезы. Зафиксировать
архитектуру и initialization, выбрать общие настройки на validation, повторить
финальную пару на трёх seeds, если позволяет бюджет. Не запускать новые RQ-family.

При необходимости ограничить tuning четырьмя конфигурациями, например
`lr in {1e-4, 3e-4}` × `dropout in {0.1, 0.2}`, с одинаковым step budget и
retrieval-based validation selection. Это предложенный бюджет, не обещание роста.
Все значения и критерии выбирать до просмотра нового test.

Hybrid sweep ограничить двумя slot budgets (например 5/10) и одним выбранным
ranker. Заранее задать primary metric и допустимую потерю aggregate, например
не более 2% относительного Recall@50 при улучшении rare-group metric. Это новое
правило для подтверждения, не основание объявить успех на уже изученных данных.

### 4. Оформить и остановить расширение scope

Достаточно main dataset и одной внешней проверки на Industrial. Beauty/другие
модели/новый большой text encoder оставить future work, если нет отдельного
обязательства воспроизвести конкретную статью. Если конечная цель именно dynamic
catalog, temporal holdout с train-only tokenizer становится обязательным; тогда
он должен заменить другие ветки, а не добавиться к ним.

Критерии завершения:

- Фиксированный протокол, воспроизводимая таблица baselines, semantic control,
  разложение retrieval/ranking, CI и ограничения.
- Один финальный hybrid: положительный, trade-off или отрицательный результат
  сохраняется без последующего подбора объяснения под желаемый ответ.
- Snapshot зависимостей, immutable run directories/manifest, команды и версии
  data/checkpoints; убрать расхождения README с реальными завершёнными запусками.
- Проверить tracked/untracked исходники: MovieLens сейчас исключён общим
  `/benchmarks/*`, а многочисленные scripts/docs ещё не закоммичены. Не удалять
  историю; явно отделить canonical pipeline от archive и smoke.
- Итоговый текст: задача и гипотезы → протокол → baseline/SID experiments →
  диагностики → ограниченное вмешательство → выводы и границы применимости.

Возможный итоговый заголовок: **«Semantic IDs для рекомендаций в разреженном
marketplace-каталоге: покрытие редких товаров, ограничения авторегрессионного
ранжирования и гибридное извлечение»**. Это формулирует проверяемую задачу без
обещания, что технология обязана победить.
