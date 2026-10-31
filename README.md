# MarketSID

MarketSID explores generative recommendations built around semantic item IDs (SIDs). The focus is on large product catalogs, especially items with few or no training interactions (tail/unseen).

The question: can product metadata help a sequential model retrieve relevant products, and where do SID generation and decoding fall short? This is research code, not a finished recommendation system.

The main question is whether product metadata can provide useful item addresses for an autoregressive recommender, and how SID construction and decoding affect retrieval across popularity groups.

Start with the [research notebook](docs/experiments/README.md) for the experiments and results. The [baseline notes](docs/experiments/results/baseline_development.md) cover SASRec and Two-Tower work. New runs can use the [experiment template](docs/experiments/protocols/experiment_template.md).

## Results

Amazon Reviews 2023: Electronics, Cell Phones and Accessories, and Office Products.
The final benchmark uses a new user cohort: **78,328 test users and 125,099 items**.
Evaluation uses the full catalog, filters each user's complete seen history, and
uses beam search 100 for generative models. Checkpoints are selected by validation Recall@50.

**Recall@50 (%), averaged over available runs.** All GenRec rows use variant C;
batch sizes below refer to training. TIGER is the repository's adapted implementation.

| Model | Trainable params (M) | All | Mid | Tail | Unseen |
|---|---:|---:|---:|---:|---:|
| TIGER · batch 1024 | 4.847 | **5.824** | 1.544 | 0.112 | 0.000 |
| GenRec · Two-Tower SID | 1.238 | 5.778 | 0.851 | 0.109 | 0.030 |
| GenRec · Two-Tower SID · batch 8192 | 1.238 | 5.560 | 0.824 | 0.109 | 0.030 |
| GenRec · BGE SID | 1.243 | 5.149 | 0.485 | 0.078 | 0.015 |
| GRU4Rec | 16.128 | 5.054 | 0.464 | 0.019 | 0.000 |
| Content Two-Tower | 0.733 | 4.888 | **1.640** | 0.774 | **0.632** |
| GenRec · BGE SID · batch 16384 | 1.243 | 4.839 | 0.349 | 0.066 | 0.000 |
| SASRec | 8.160 | 4.378 | 0.124 | 0.011 | 0.000 |
| LightGCN | 13.019 | 3.513 | 0.567 | 0.109 | 0.000 |
| ItemKNN | — | 0.651 | 0.840 | **0.811** | 0.000 |

Parameter counts cover the recommender only, excluding preprocessing models
(BGE, RQ-VAE, and the SID teacher) and cached features; shared weights count once.
ItemKNN stores a similarity matrix rather than neural parameters.

- **TIGER and GenRec with Two-Tower SIDs lead overall.** Their Recall@50 difference
  is small: 36 hits, with a paired 95% interval spanning zero. GenRec achieves this
  with **3.9× fewer trainable parameters** (1.24M versus TIGER's 4.85M).
- **Two-Tower embeddings helped SID construction:** GenRec's overall Recall@50
  is about 12.2% higher than with BGE SIDs. Larger training batches did not improve it.
- **Rare-item quality remains a limitation.** Content Two-Tower leads mid/unseen;
  ItemKNN has the highest observed tail recall. GenRec's overall advantage does not
  establish an advantage on rare items.

See the [benchmark protocol](docs/experiments/protocols/final_benchmark.md) and
[full report](artifacts/benchmarks/final-main/reports/comparison_final_20261030.md)
for Recall/NDCG@10/20/50 and paired comparisons. The full report is a local generated
artifact; the table above is retained in Git.

## Approach

```text
Product metadata → BGE text embedding → RQ-VAE → semantic item ID
                                                       ↓
User history → Transformer encoder → autoregressive SID decoder
                                                       ↓
                                      Catalog-constrained beam search
                                                       ↓
                                         Filter seen items → recommendations
```

RQ-VAE encodes each item as a short sequence of discrete tokens. GenRec predicts those tokens from a user's history. A catalog trie limits decoding to valid item IDs.

SASRec and a content-based Two-Tower model are included as baselines. Two-Tower item vectors can also be used to train the SID tokenizer.

## Setup

Run everything from the repository root in a virtual environment. Install a CPU or CUDA build of PyTorch first. Development used Python 3.14; dependencies are not fully pinned.

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for your machine, then the remaining Python dependencies.
python -m pip install numpy polars scipy scikit-learn sentence-transformers \
  "datasets==3.6.0" huggingface-hub pyarrow matplotlib
```

Preprocessing uses Hugging Face dataset scripts with `trust_remote_code=True` and `datasets==3.6.0`. CUDA is recommended for training and full-catalog evaluation, though most scripts support CPU.

## Main dataset

The main pipeline combines three categories from the `Amazon Reviews 2023` dataset:

- Electronics
- Cell Phones and Accessories
- Office Products

The pipeline takes 5-core rating data, filters and samples users, applies iterative k-core filtering, and splits each user's history chronologically (leave-one-out): last event for test, previous event for validation, earlier events for training. Ratings count as implicit interactions; there is no global date cutoff.

The original exploratory snapshot has **80,471 users**, **134,288 items**, and **574,699 training examples**; the final benchmark above uses a separate cohort. Regenerating the dataset may change these counts. Item ID 0 is reserved for padding.

Popularity groups use training interaction counts:

| Group | Training interactions per item |
|---|---:|
| Head | >20 |
| Mid | 6–20 |
| Tail | 1–5 |
| Unseen | 0 |

The tokenizer uses metadata for the whole retained catalog, including held-out items. **Unseen** means no training interactions, not missing metadata.

### Prepare data and text embeddings

```bash
python -m scripts.prepare_data
python -m scripts.prepare_metadata
python -m scripts.prepare_model_data
python -m scripts.encode_items
```

Outputs go to `data/processed/`: indexed items, interactions, sequences, and text embeddings built from titles, categories, features, and descriptions.

Text is encoded with `BAAI/bge-small-en-v1.5` (384 dimensions, max 512 tokens). Cache metadata tracks the encoder settings and source catalog.

### Content SID baseline

```bash
python -m scripts.train_rqvae
python -m scripts.export_semantic_ids
python -m scripts.train_genrec --variant C --sid-source content
```

Content SIDs use three RQ codes and a collision suffix (4th token) to make each item's four-token ID unique. GenRec C initializes from the RQ codebooks and ties semantic input/output embeddings.

You can try a small validation run:

```bash
python -m scripts.evaluate_genrec --sid-source content --split valid \
  --beam-size 100 --batch-size 10 --limit 10 --diagnose-prefixes
```

Remove `--limit` for full validation. After choosing a configuration, run test evaluation:

```bash
python -m scripts.evaluate_genrec --sid-source content --split test \
  --beam-size 100 --batch-size 64 --diagnose-prefixes
```

Evaluation reports Recall/NDCG@10/20/50 by popularity group. It searches the retained catalog without sampled negatives. Beam search is approximate, and filtering seen items can leave fewer than 50 results.

### Recommendation baselines and Two-Tower SIDs

```bash
python -m scripts.train_sasrec
python -m scripts.train_two_tower
```

For Two-Tower-based SIDs, train both baselines, then export item vectors and retrain the tokenizer and GenRec:

```bash
python -m scripts.export_two_tower_embeddings
python -m scripts.train_twotower_rqvae
python -m scripts.export_twotower_semantic_ids
python -m scripts.train_genrec --variant C --sid-source twotower
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes
```

Two-Tower computes embeddings from product content, including for items without a trained item-ID embedding.

## Model and tokenizer experiments

`--variant` sets up GenRec's semantic embeddings and output configuration:

| Variant | Semantic embeddings | Semantic output weights |
|---|---|---|
| A | Learned from scratch | Separate |
| B | Initialized from RQ codebooks | Separate |
| C | Initialized from RQ codebooks | Tied to input embeddings |
| D | Frozen RQ base + trainable, regularized delta | Tied to input embeddings |

The main defaults are in [`scripts/train_genrec.py`](scripts/train_genrec.py): 50 history items, model width 128, 4 attention heads, 3 encoder layers, 2 decoder layers, 0.2 dropout, batch size 2048, and learning rate `3e-4`. Checkpoints are chosen by validation token loss. The main evaluator uses GenRec C.

`--sid-source` chooses how Semantic IDs are built:

| SID source | Tokenizer input | Additional supervision | Structure |
|---|---|---|---|
| `content` | BGE embeddings | None | 3 RQ + collision suffix |
| `twotower` | Two-Tower item vectors | None | 3 RQ + collision suffix |
| `rq4_a` | Two-Tower item vectors | None | 4 RQ |
| `rq4_b` | BGE embeddings | Predict Two-Tower vectors from the final RQ sum | 4 RQ |
| `rq4_c` | BGE embeddings | Predict Two-Tower vectors at each partial RQ depth | 4 RQ |

For four-RQ SIDs, export assigns distinct final codewords within each three-code bucket using minimum-cost matching. Buckets must fit in the last codebook. A changed catalog can change the assignments.

Tokenizer A/B/C labels are unrelated to GenRec variants A/B/C/D. See [SID ablations](docs/experiments/protocols/sid_ablation.md) for more information.

Use each SID file with its matching RQ checkpoint. GenRec best checkpoints use `checkpoints/genrec_<sid_source>_<variant>_best.pt` and are overwritten when rerun. Histories are timestamped. A new SID mapping requires retraining GenRec.

## Diagnostics and current research

Results still favor popular items. Tokenizer changes have not fixed the head/tail gap. The main diagnostics identify where targets drop out during decoding and how training frequency affects path scores.

- **Prefix survival:** `--diagnose-prefixes` records whether the true target
  prefix survives each beam-pruning step, before seen-item filtering.
- **Path likelihood:** `--diagnose-paths` scores every true SID with teacher
  forcing and relates token/prefix probabilities to training frequencies.
  `--shuffle-histories` adds a comparison using another user's history.
- **Prior correction:** `--prior-alpha-sweep` evaluates decoding scores that
  subtract a weighted, smoothed conditional training log prior.
- **Trie normalization:** `--trie-normalize` normalizes probabilities over valid
  catalog children at each prefix, instead of the full token vocabulary.

For example, compare normal decoding with trie normalization on validation:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes --trie-normalize
```

For a short run, add `--limit 100` to both commands. The [decoding guide](docs/experiments/protocols/sid_path_likelihood.md) explains prior sweeps and diagnostic outputs in `artifacts/`.

## Separate benchmarks

The [main-dataset comparison](benchmarks/main-comparison/README.md) runs ItemKNN, GRU4Rec, LightGCN, and an adapted TIGER alongside GenRec under a shared evaluation protocol.

Other datasets have separate data, tokenizer, model, and evaluation pipelines:

- [Amazon Beauty and Personal Care](benchmarks/amazon-beauty/README.md)
- [Amazon Industrial and Scientific](benchmarks/amazon-industrial-scientific/README.md)
- [MovieLens 1M](benchmarks/movielens-1m/README.md)

Read each benchmark's protocol before comparing scores. The Amazon benchmarks filter duplicate user/item pairs and products without usable metadata differently from the main pipeline.

## Repository layout

```text
src/data/               Datasets, preprocessing helpers, embedding cache
src/models/             Recommenders, RQ-VAE variants, trie, beam search
scripts/                Main-dataset preparation, training, evaluation, diagnostics
benchmarks/             Separate dataset pipelines
docs/                   Documentation index and repository layout
docs/experiments/       Protocols, research log, reviews, and interpreted results
tests/                  Existing unit tests
data/                   Raw and processed datasets, embeddings
checkpoints/            Main-dataset model weights
artifacts/main/         Training histories, evaluations, diagnostics, and reports
artifacts/benchmarks/   Results grouped by benchmark dataset
artifacts/archive/      Historical runs and source snapshots
```

See the [docs index](docs/README.md) and [directory guide](docs/repository_layout.md). Benchmark weights are under `benchmarks/<dataset>/checkpoints/`; raw outputs are under `artifacts/`. Generated data, weights, and most raw artifacts are Git-ignored. Environment packaging and broader validation are still open.

Run the existing lightweight tests with:

```bash
python -m unittest discover -s tests
```

# Основные наблюдения и выводы проекта

Это выводы из проведённых экспериментов, а не универсальные свойства Generative Recommendation models.

1. **Перед RQ стоит попробовать эмбеддинги обученного Content Two-Tower вместо исходных BGE (или другого энкодера).** В моем сравнении GenRec C общий Recall@50 вырос с **5.31% до 5.67%**. Но выигрыш пришёл преимущественно от head: mid/tail не улучшились. Это полезная практическая возможность, а не гарантия лучших SID для любой задачи. [Результаты](results/overview.md#main-sid-comparison).

2. **Выученное соответствие «товар → SID» действительно значительно помогает GenRec.** После перемешивания целых SID между товарами близкой популярности validation Recall@50 упал с **6.61% до 3.45%**. Значит, исходные адреса несут полезную структуру. Контроль проведён с одной парой seed и проверяет структуру вместе с её соответствием RQ-инициализации. [Контроль](results/sid_permutation_results_20261018.md).

3. **Высокий общий Recall не означает хорошее качество на редких товарах.** В полном validation-сравнении GenRec опережал Content Two-Tower по общему Recall@50, но проигрывал ему на mid, tail и unseen. Гипотеза о преимуществе GenRec на редких товарах пока не подтвердилась. [Сравнение](results/full_tt_control_20261018.md).

4. **GenRec умеет находить области с редкими товарами, но часто теряет сами товары.** Правильная история давала заметно лучшее покрытие префиксами, чем перемешанная история и популярностные контроли. Однако короткие префиксы оставляют много кандидатов для ранжирования, а длинные исключают большинство редких целей. Проблемы есть на обоих этапах. [Диагностика](results/prefix_retrieval_full_valid_20261017.md).

5. **Семантические префиксы полезнее там, где есть обучающий сигнал от соседних товаров.** В отдельных группах с высокой поддержкой других товаров внутри префикса ограничение кандидатов помогало. В целом потерь было больше, чем выигрышей. Это наблюдаемая зависимость, а не доказательство, что увеличение групп само по себе улучшит рекомендации. [Разбор](results/prefix_conditions_20261018.md).

6. **Более сложный tokenizer не обязательно даёт лучшие рекомендации.** Четыре RQ-кода с уникальным назначением и поведенческое обучение на нескольких глубинах не превзошли простой Two-Tower → 3 RQ + collision по общему Recall@50. Замены collision-токена оказалось недостаточно. [Абляции](results/overview.md#main-sid-comparison).

7. **Коррекция scores и гибриды дают компромисс, а не бесплатное улучшение.** Частотная коррекция увеличивала некоторые rare-item попадания ценой общего качества. В гибридах простое добавление товаров полного Two-Tower не уступало ограничению его кандидатов префиксами по tail/unseen. Поэтому пользу гибрида нельзя приписывать самим префиксам. [Коррекция](results/overview.md#decoding-interventions), [гибриды](results/full_tt_control_20261018.md).

8. **Сложность данных важна, но достижение потолка не доказано.** На MovieLens и Industrial абсолютные метрики значительно выше. Однако отличаются данные, каталоги и настройки; это не изолированный эксперимент над разреженностью. Сравнивать модели нужно внутри одного протокола. [Другие датасеты](results/overview.md#separate-benchmarks).

