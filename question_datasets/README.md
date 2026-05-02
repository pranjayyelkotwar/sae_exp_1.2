## Overview
Compact reference for question dataset module: base class -> specialized datasets -> combined dataset. Shows original vs transformed fields, unified schema, mappings, and usage.

## Architecture
- `BaseQuestionDataset`: loads HF dataset, filters, normalizes, encodes prompt.
- Specialized datasets: `ARCEasyDataset`, `MMLUDataset`, `HLEDataset` implement `normalize_record`.
- `CombinedQuestionDataset`: concatenates datasets, adds `combined_dataset_idx`.

## Dataset Schemas

ARCEasy

Original fields

| field | notes |
|---|---|
| `question` | text |
| `choices` | dict with `label`, `text` |
| `answerKey` | optional |
| `id` | optional |

Transformed fields

| field | source |
|---|---|
| `prompt_text` | built from `question` + ordered `choices` |
| `question_text` | `question` |
| `choices` | ordered list from `choices.label/text` |
| `gold_answer` | `answerKey` |
| `subject` | `science` (constant) |

MMLU

Original fields

| field | notes |
|---|---|
| `question` | text |
| `choices` | list or None |
| `answer` | index or letter |
| `subject` | optional |

Transformed fields

| field | source |
|---|---|
| `prompt_text` | built from `question` + `choices` |
| `question_text` | `question` |
| `choices` | `choices` |
| `gold_answer` | normalized `answer` (letter) |
| `subject` | `subject` or `unknown` |

HLE

Original fields

| field | notes |
|---|---|
| `question` | text |
| `choices` / `options` / `multiple_choice_targets` | optional lists |
| `answer` / `answer_key` / `correct_answer` / `solution` | optional |
| `question_id` / `id` / `uid` | optional |
| `subject` / `category` / `raw_subject` | optional |
| `image` | may be empty string to keep example |

Transformed fields

| field | source |
|---|---|
| `prompt_text` | built from `question` + choices |
| `question_text` | `question` |
| `choices` | best-available choices field |
| `gold_answer` | first available answer field |
| `source_id` | first available id field |
| `question_type` | `answer_type` or inferred |

## Unified Schema (CombinedQuestionDataset)

| field |
|---|
| `prompt_text` |
| `question_text` |
| `difficulty_label` |
| `source_dataset` |
| `source_split` |
| `source_id` |
| `subject` |
| `question_type` |
| `choices` |
| `gold_answer` |
| `has_image` |
| `token_count` |
| `combined_dataset_idx` |

## Field Mapping (dataset → unified)

- ARCEasy: `question`→`question_text`, `choices.label/text`→`choices`, `answerKey`→`gold_answer`, `id`→`source_id`, constant `science`→`subject`, `dataset_name`→`source_dataset`.
- MMLU: `question`→`question_text`, `choices`→`choices`, `answer`→`gold_answer` (index→letter), `subject`→`subject`, `dataset_name`→`source_dataset`.
- HLE: `question`→`question_text`, `choices|options|multiple_choice_targets`→`choices`, first of `answer|answer_key|correct_answer|solution`→`gold_answer`, `question_id|id|uid`→`source_id`, `dataset_name`→`source_dataset`.

## Usage

```python
from question_datasets.combined import build_combined_question_dataset

combined = build_combined_question_dataset(
    dataset_names=["arc_easy","mmlu","hle"],
    tokenizer=my_tokenizer,
    max_token_length=512,
)
prompt_tokens, idx, seq_len, metadata = combined[0]
```
