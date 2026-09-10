# LKValues

**LKValues** is a survey-grounded Sinhala–English resource suite for studying the alignment of large language models with Sri Lankan societal values.

📄 **Paper:** [LKValues: Aligning Large Language Models with Sri Lankan Societal Values](https://arxiv.org/abs/2607.20410)

🤗 **Datasets:** [Nethmi14/LKValues on Hugging Face](https://huggingface.co/datasets/Nethmi14/LKValues)

The project includes:

- **LKvaluesIT** — a bilingual instruction-tuning dataset for value-grounded generation.
- **LKvaluesBench** — a bilingual benchmark for value-sensitive judgment.

> **Important:** LKValues is a survey-bounded, descriptive research resource. It is not an official, exhaustive, or authoritative account of Sri Lankan values.

---

## Resources

The full LKValues datasets are publicly available on Hugging Face:

🤗 **[Nethmi14/LKValues](https://huggingface.co/datasets/Nethmi14/LKValues)**

### LKvaluesIT

**LKvaluesIT** is a bilingual instruction-tuning dataset designed for **value-grounded generation and supervised fine-tuning**.

Given a situation and a target Sri Lankan societal value, the model is expected to generate a short explanation describing how the situation supports that value.

The public release contains the full bilingual instruction datasets:

| Language | Instances |
| --- | ---: |
| English | ~150,000 |
| Sinhala | ~150,000 |
| Total | ~300,000 |

Each example contains:

- a short situation or scenario;
- a target societal value;
- a value-grounded explanation.

The English and Sinhala datasets are aligned bilingual versions of the same instruction resource.

### LKvaluesBench

**LKvaluesBench** is a bilingual evaluation benchmark designed for **controlled value-sensitive judgment**.

It contains **1,000 aligned benchmark instances** in English and Sinhala.

Each item contains:

- a question or scenario;
- `Statement_A`;
- `Statement_B`;
- a gold label from `A`, `B`, `BOTH`, or `0`;
- a mapped primary societal value.

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

## Models Evaluated

The paper evaluates a range of proprietary and open-weight models and adapts the following base models using LKvaluesIT:

- Qwen3.5-4B-Base
- Qwen3.5-9B-Base
- Aya-Expanse-8B-Base

The experiments show substantial improvements for the evaluated Qwen-family models in Sinhala and English, while gains remain dependent on model family, adaptation method, and training configuration.

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
