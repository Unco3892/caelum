# Asset sources and licenses

CAELUM's original code and software documentation use the [MIT License](../LICENSE).
Third-party code, datasets, and model weights retain their upstream terms. The
links below identify those sources; CAELUM's license does not relicense them.

## Models and encoders

| Asset | Upstream license | Primary record |
|---|---|---|
| Qwen/Qwen3-8B | apache-2.0 | [Official card](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen/Qwen3-14B | apache-2.0 | [Official card](https://huggingface.co/Qwen/Qwen3-14B) |
| Qwen/Qwen3-32B | apache-2.0 | [Official card](https://huggingface.co/Qwen/Qwen3-32B) |
| mistralai/Ministral-3-8B-Instruct-2512 | apache-2.0 | [Official card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) |
| google/gemma-4-E4B-it | apache-2.0 | [Official card](https://huggingface.co/google/gemma-4-E4B-it) |
| microsoft/phi-4 | mit | [Official card](https://huggingface.co/microsoft/phi-4) |
| Qwen/Qwen3.5-35B-A3B | apache-2.0 | [Official card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| sentence-transformers/all-mpnet-base-v2 | apache-2.0 | [Official card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| sentence-transformers/all-MiniLM-L12-v2 | apache-2.0 | [Official card](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) |
| BAAI/bge-large-en-v1.5 | mit | [Official card](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| intfloat/e5-large-v2 | mit | [Official card](https://huggingface.co/intfloat/e5-large-v2) |

The hosted judge, Gemini 2.5 Flash-Lite, is accessed through the Gemini Developer
API under the [Gemini API terms](https://aistudio.google.com/docs/terms) and
[Google APIs terms](https://developers.google.com/terms).

## Datasets

| Asset | Upstream license statement | Primary record and limitation |
|---|---|---|
| TriviaQA | Apache-2.0 for upstream code/data release | [Official repository](https://github.com/mandarjoshi90/triviaqa); [official site](https://nlp.cs.washington.edu/triviaqa/) explicitly says UW does not own copyright in included questions/documents. Retain source provenance and notices. |
| OpenNQ / Natural Questions | Official repository Apache-2.0 | [Official repository](https://github.com/google-research-datasets/natural-questions). Preserve inherited Wikipedia attribution and terms where applicable; curated release provenance remains relevant. |
| SimpleQA | MIT, explicitly listed for this dataset | [Official release](https://github.com/openai/simple-evals#evals). Preserve MIT notice. |
| PopQA | MIT repository release | [Author repository](https://github.com/AlexTMallen/adaptive-retrieval). Preserve Wikidata and pageview provenance; do not invent a separate CC license for the author HF card. |
| AmbigQA / AmbigNQ | CC BY-SA 3.0 Unported | The LICENSE inside the [official data download](https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip) and the [author dataset card](https://huggingface.co/datasets/sewon/ambig_qa) specify this license. Credit the creators, link the license, identify adaptations, and retain the applicable ShareAlike terms. |
| AmbigInst and curated spectral inputs | Spectral release expressly states MIT | [Official spectral repository](https://github.com/MLO-lab/spectral_uncertainty_decomposition#license). This statement covers the release as offered by that distributor; it cannot erase inherited rights. |
| CondAmbigQA | Apache-2.0 declared by the distributor | The [CondAmbigQA-2K card](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA-2K) identifies the release loaded by `src/prepare_datasets.py`. This is a legacy input outside the six reported datasets. The release license does not replace terms for inherited source text. |

## Upstream code

- [Spectral uncertainty decomposition](https://github.com/MLO-lab/spectral_uncertainty_decomposition), MLO Lab. [Repository license](https://github.com/MLO-lab/spectral_uncertainty_decomposition/blob/main/LICENSE).
- [CLEAR](https://github.com/Unco3892/clear), Ilia Azizi. [Repository license](https://github.com/Unco3892/clear/blob/main/LICENSE.txt).
