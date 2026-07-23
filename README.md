# LKValues

**LKValues** is a survey-grounded Sinhala–English resource suite for studying the alignment of large language models with Sri Lankan societal values.

The project includes:

- **LKvaluesIT** — a bilingual instruction-tuning dataset for value-grounded generation.
- **LKvaluesBench** — a bilingual benchmark for value-sensitive judgment.
- Supporting prompts, value definitions, evaluation scripts, and documentation.

> **Important:** LKValues is a survey-bounded, descriptive research resource. It is not an official, exhaustive, or authoritative account of Sri Lankan values.

---

## Overview

Many existing LLM alignment datasets and benchmarks are dominated by English-language and Western cultural assumptions. LKValues focuses on Sri Lanka, a multilingual, multi-ethnic, and multi-religious society that remains underrepresented in current value-alignment research.

The resource construction process begins with a survey presented simultaneously in **Sinhala, Tamil, and English** to **205 Sri Lankan respondents**. From 51 candidate constructs, we retain **40 majority-endorsed societal values** under the survey's sampling and measurement conditions.

These values are then used to construct two bilingual Sinhala–English resources.

---

## Resources

### LKvaluesIT

LKvaluesIT is a bilingual instruction-tuning dataset derived from Sri Lankan news published between 2009 and 2023.

Each example contains:

- a short situation or scenario;
- a target societal value;
- a value-grounded explanation;
- an English version and an aligned Sinhala version.

The release contains approximately:

| Split | English | Sinhala |
|---|---:|---:|
| Train | 120,000 | 120,000 |
| Validation | 15,000 | 15,000 |
| Test | 15,000 | 15,000 |

This corresponds to approximately **150,000 aligned bilingual pairs**.

LKvaluesIT is designed for value-grounded generation and bilingual instruction tuning. It should not be interpreted as a complete representation of all Sri Lankan communities or viewpoints.

### LKvaluesBench

LKvaluesBench is a bilingual value-sensitive judgment benchmark containing **1,000 aligned items** in Sinhala and English.

Each item presents:

- a question or scenario;
- `Statement_A`;
- `Statement_B`;
- a gold label from `A`, `B`, `BOTH`, or `0`;
- a mapped primary societal value.

The benchmark combines:

- 491 human-curated items adapted from SinhalaMMLU; and
- 509 additional scenario-based items generated with human-in-the-loop verification.

---

## Dataset Structure

The repository is organized as follows:

```text
LKValues/
├── README.md
├── DATA_CARD.md
├── CITATION.cff
├── LICENSE-DATA
├── LICENSE-CODE
│
├── data/
│   ├── lkvalues_it/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   │
│   ├── lkvalues_bench/
│   │   ├── lkvaluesbench_en.jsonl
│   │   └── lkvaluesbench_si.jsonl
│   │
│   └── metadata/
│       ├── value_inventory.csv
│       ├── dataset_statistics.json
│       └── data_schema.md
│
├── examples/
│   ├── lkvalues_it_examples.jsonl
│   └── lkvalues_bench_examples.jsonl
│
├── prompts/
│   ├── value_tagging_prompt.txt
│   ├── scenario_extraction_prompt.txt
│   └── benchmark_evaluation_prompts.txt
│
├── scripts/
│   ├── load_dataset.py
│   ├── validate_schema.py
│   └── evaluation/
│
└── docs/
    ├── annotation_guidelines.md
    ├── dataset_construction.md
    ├── survey_instrument/
    └── release_notes.md
```

The final public structure may change slightly as the release is finalized.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/LKValues.git
cd LKValues
```

Load a JSONL file in Python:

```python
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

    return records


examples = load_jsonl(
    "data/lkvalues_bench/lkvaluesbench_en.jsonl"
)

print(f"Loaded {len(examples)} examples")
print(examples[0])
```

---

## Benchmark Evaluation

For LKvaluesBench, models must return exactly one label:

```text
A
B
BOTH
0
```

The labels mean:

| Label | Meaning |
|---|---|
| `A` | Only Statement A is justifiable |
| `B` | Only Statement B is justifiable |
| `BOTH` | Both statements are justifiable |
| `0` | Neither statement is justifiable |

Outputs that cannot be normalized to one of these four labels should be counted as invalid.

Evaluation scripts and prompts will be released under `scripts/evaluation/` and `prompts/`.

---

## The 40 Societal Values

The retained value inventory includes constructs related to family, respect, compassion, responsibility, justice, multiculturalism, resilience, environmentalism, accountability, education, spirituality, political freedom, and other social and civic concerns.

The complete inventory, definitions, endorsement statistics, and related terminology are provided in:

```text
data/metadata/value_inventory.csv
```

These values were retained based on majority endorsement within the collected survey sample. They should not be treated as universally endorsed by all Sri Lankans.

---

## Construction Pipeline

The main stages are:

1. Selection of value-related items from established international frameworks.
2. LLM-assisted surfacing of additional candidate constructs for contextual coverage.
3. Manual consolidation and survey operationalization.
4. Majority-endorsement analysis using responses from 205 participants.
5. Value tagging of Sri Lankan news records.
6. Scenario extraction and Sinhala translation.
7. Human validation of value labels, explanations, translations, and benchmark items.
8. Model fine-tuning and bilingual evaluation.

LLMs were used as scalable processing tools. They did not determine the final retained value inventory. Final retention was based on survey responses.

---

## Scope and Limitations

LKValues has several important limitations:

- The survey sample is not demographically balanced across all Sri Lankan communities.
- Majority-based retention may underrepresent minority-held, contested, or polarizing values.
- The current datasets cover Sinhala and English, while Tamil was included only in the survey stage.
- The news sources and model-assisted processing pipeline may introduce editorial, linguistic, political, religious, or demographic biases.
- LKvaluesIT currently focuses on positive value-support explanations and does not fully represent value conflicts or conditional judgments.
- The benchmark should be used as a research instrument, not as a normative standard for Sri Lanka.

Researchers should report these limitations when using the resources.

---

## Responsible Use

LKValues is intended for:

- multilingual NLP research;
- low-resource language evaluation;
- culturally grounded model analysis;
- value-alignment research;
- instruction tuning and benchmarking.

It should **not** be used to:

- define an official Sri Lankan value system;
- make decisions about individuals or communities;
- stereotype Sri Lankan ethnic, linguistic, religious, or social groups;
- justify discriminatory or coercive model behavior;
- claim that all Sri Lankans share the same values.

Users should conduct subgroup-sensitive and harm-aware evaluations before deploying models trained with this resource.

---

## Data Release and Privacy

The public release does not include:

- raw survey responses;
- participant names or direct identifiers;
- private demographic combinations that may enable re-identification;
- original copyrighted news articles;
- API keys, credentials, or internal project materials.

Only derived annotations, paraphrased scenarios, aligned translations, metadata, and benchmark items are released.

---

## Models Evaluated

The paper evaluates a range of proprietary and open-weight models and adapts the following base models using LKvaluesIT:

- Qwen3.5-4B-Base
- Qwen3.5-9B-Base
- Aya-Expanse-8B-Base

The experiments show substantial improvements for the evaluated Qwen-family models in Sinhala and English, while gains remain dependent on model family, adaptation method, and training configuration.

---

## Paper

**LKValues: Aligning Large Language Models with Sri Lankan Societal Values**

Paper link: [**LKValues**](https://arxiv.org/abs/2607.20410)

arXiv: **Coming soon**
---

## Licenses

- Dataset and documentation: **Creative Commons Attribution 4.0 International**
- Code: license details will be provided in `LICENSE-CODE`

Users must also follow the licenses and terms of any third-party datasets or models used alongside LKValues.

---

## Contact

For questions about the dataset or paper, contact:

- **Nethmi Muthugala** — `neth.muthugala1@gmail.com`
- **Deyi Xiong** — `dyxiong@tju.edu.cn`

---

## Acknowledgements

This research was supported by the National Key Research and Development Program of China (Grant No.~2024YFE0203000). We sincerely thank all survey participants for contributing their time and perspectives. We are also grateful to the annotators, proofreaders, and Sri Lankan professionals who supported the survey design, data validation, linguistic review, cultural verification, and overall development of the LKValues resources.
