# MM Reranker Eval

A unified, elegant package for evaluating multimodal reranking models across various retrieval tasks.

## Features

- **Unified Interface**: Single API for all retrieval tasks (txt2img, img2txt, mix2mix, etc.)
- **Flexible Modality Support**: Text, image, and extensible to video and other modalities
- **Multiple Models**: Easy integration of different reranker models
- **Parallel GPU Evaluation**: Automatic distribution across multiple GPUs
- **Comprehensive Metrics**: Recall@K, MRR, NDCG@K, and extensible
- **YAML Configuration**: Clean configuration management
- **CLI Support**: Command-line interface for easy evaluation

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Create virtual environment

Choose your environment path and create it:

```bash
bash ./scripts/setup_env.sh /your-path/envs/mm-reranker/dev
```

### 3. Activate environment

```bash
source scripts/activate_env.sh /your-path/envs/mm-reranker/dev
```

### 4. Install package

```bash
# Basic installation
uv sync --active

# With flash attention (recommended for better performance)
uv sync --active --no-build-isolation --extra flash-attention

# With ColQwen2 support (requires colpali-engine)
uv sync --active --extra colpali

# With development tools
uv sync --active --extra dev

# or
uv sync --active --no-build-isolation --extra flash-attention --extra colpali --extra dev
```

**Note for ColQwen2 users:**
- ColQwen2 requires `colpali-engine >= 0.3.4` and `transformers > 4.46.1`
- If the PyPI version is outdated, install from source:
  ```bash
  pip install git+https://github.com/illuin-tech/colpali
  ```

**Note for MonoQwen2-VL users:**
- MonoQwen2-VL requires `peft >= 0.7.0` (for LoRA adapter support)
- This is automatically installed with the base package
- The model is based on Qwen2-VL-2B-Instruct with LoRA adapters

### 5. Add and remove package

```bash
uv add --active [package-name]
uv remove --active [package-name]
```

### 6. Deactivate environment

```bash
deactivate
```

### 7. Switch to different environment

```bash
# Deactivate current environment
deactivate

# Activate another environment
source scripts/activate_env.sh /path/to/another/env
```

## Supported Models

The package currently supports the following reranker models:

- **Jina AI Reranker M0** (`jinaai/jina-reranker-m0`): Cross-encoder based multimodal reranker
- **Jina CLIP v2** (`jinaai/jina-clip-v2`): Embedding-based CLIP model for retrieval
- **DSE Qwen2 MRL** (`MrLight/dse-qwen2-2b-mrl-v1`): Qwen2VL-based multimodal embedding model
- **BGE-VL-MLLM** (`BAAI/BGE-VL-MLLM-S1`): BAAI's multimodal reranker supporting comprehensive multimodal retrieval
- **GME-Qwen2-VL** (`Alibaba-NLP/gme-Qwen2-VL-7B-Instruct`): Alibaba's General Multimodal Embedding model based on Qwen2-VL
- **ColQwen2** (`vidore/colqwen2-v1.0`): ColPali-based vision-language model for document retrieval using multi-vector embeddings
- **MonoQwen2-VL** (`lightonai/MonoQwen2-VL-v0.1`): Pointwise reranker based on Qwen2-VL-2B-Instruct that uses True/False generation for relevance scoring (requires `peft` for LoRA adapter support)
- **GLM-4.1V-9B-Thinking** (`THUDM/GLM-4.1V-9B-Thinking` or `zai-org/GLM-4.1V-9B-Thinking`): Native thinking VLM with 3 scoring modes: `logits_pyes` (P(Yes) from token logits, fast), `generative_ranking` (generate ranking sequence, interpretable), `listwise_scores` (generate score list 0-1, fine-grained)

## Quick Start

### 1. Programmatic Usage

```python
from mm_reranker_eval import MMReranker, Query, Document

# Initialize reranker with remote model name
reranker = MMReranker("jinaai/jina-reranker-m0", device="cuda")
# Or use other supported models:
# reranker = MMReranker("jinaai/jina-clip-v2", device="cuda")  # Jina CLIP v2
# reranker = MMReranker("MrLight/dse-qwen2-2b-mrl-v1", device="cuda")  # DSE Qwen2 MRL
# reranker = MMReranker("BAAI/BGE-VL-MLLM-S1", device="cuda")  # BGE-VL-MLLM
# reranker = MMReranker("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct", device="cuda")  # GME-Qwen2-VL
# reranker = MMReranker("vidore/colqwen2-v1.0", device="cuda")  # ColQwen2
# reranker = MMReranker("lightonai/MonoQwen2-VL-v0.1", device="cuda")  # MonoQwen2-VL
# reranker = MMReranker("THUDM/GLM-4.1V-9B-Thinking", device="cuda", scoring_mode="logits_pyes")  # GLM4V Thinking

# Or use a local model path (type auto-detected from directory name or config.json)
# reranker = MMReranker("/path/to/local/model", device="cuda")
# reranker = MMReranker("./models/my-model", device="cuda")

# Create query and documents
query = Query(text="slm markdown")
documents = [
    Document(image="https://example.com/image1.png"),
    Document(image="https://example.com/image2.png"),
    Document(text="A document about markdown"),
]

# Rank documents
# The reranker automatically infers:
#   - query_type="text" (pure text query)
#   - doc_type="image" for image documents
#   - doc_type="text" for text documents
result = reranker.rank(query, documents)
print(f"Ranked indices: {result.ranked_indices}")
print(f"Scores: {result.scores}")
```

**Note:** Models like Jina reranker require both `query_type` and `doc_type` to handle different retrieval scenarios correctly. The framework automatically infers these types based on the query and document modalities, ensuring optimal model performance.

### 2. CLI Usage

Create a configuration file:
```bash
mmranker create-config -o my_eval.yaml
```

Edit the configuration file:
```yaml
model:
  # Remote model name or local path (auto-detected)
  name: jinaai/jina-reranker-m0  # or /path/to/model
  device: cuda
  num_gpus: 2
  use_flash_attention: true

data:
  eval_data_path: data/eval.jsonl
  candidate_docs_path: data/candidates.jsonl
  max_queries: 100

metrics:
  recall_k: [1, 3, 5, 7, 10]
  ndcg_k: [5, 10]

output_dir: results/my_experiment
rank_kwargs:
  max_length: 2048
save_per_query: true
```

Run evaluation:
```bash
mmranker eval my_eval.yaml
```

## Data Format

### Evaluation Data (JSONL)

Each line contains a query and its matching document:

**Text-to-Image:**
```json
{"query": "Sayulita beach town", "match": "images/beach.jpg", "dataset": "VisualNews", "id": 1}
```

**Image-to-Text:**
```json
{"query": "images/soldier.jpg", "match": "A South Korean soldier patrols", "dataset": "VisualNews", "id": 2}
```

**Mixed Modality:**
```json
{
  "query": {"text": "Who is the developer?", "image": "images/aircraft.jpg"},
  "match": {"text": "Tupolev Tu-154. The Tupolev...", "image": "images/tupolev.jpg"},
  "dataset": "INFOSEEK",
  "id": 3
}
```

### Candidate Documents (JSONL)

Each line is a candidate document:
```json
{"text": "A text document"}
{"image": "path/to/image.jpg"}
{"text": "Caption for image", "image": "path/to/image.jpg"}
```

## Architecture

```
mm_reranker_eval/
├── data/
│   └── types.py          # Core data types (Query, Document, EvalSample)
├── reranker/
│   ├── base.py           # BaseReranker abstract class
│   ├── jina.py           # Jina reranker implementation
│   └── factory.py        # MMReranker factory
├── evaluation/
│   ├── metrics.py        # Evaluation metrics
│   └── evaluator.py      # Parallel evaluator
├── config.py             # YAML configuration handling
└── cli.py                # Command-line interface
```

## Local Model Support

The package supports both remote models (e.g., from HuggingFace) and local model paths with **automatic type detection**.

### Using Local Models

```python
# Download and save a model locally first
from transformers import AutoModel
model = AutoModel.from_pretrained("jinaai/jina-reranker-m0", trust_remote_code=True)
model.save_pretrained("./models/my-model")

# Then use the local path - model type is automatically detected
from mm_reranker_eval import MMReranker
reranker = MMReranker("./models/my-model", device="cuda")
```

### How Auto-Detection Works

For local paths, the model type is detected from:
1. **Directory name patterns** - e.g., directories containing "jina" are recognized as Jina models
2. **Config files** - reads `config.json` to identify model from `model_type`, `_name_or_path`, or `architectures`
3. **Registry patterns** - checks against registered model patterns

## Extending the Package

### Adding a New Model

The package uses a **template method pattern** - the base class handles the ranking flow, you just implement model-specific details:

```python
from mm_reranker_eval.reranker import BaseReranker, register_reranker
from mm_reranker_eval.data.types import Query, Document, Modality
from typing import List, Set

class MyReranker(BaseReranker):
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        # Load your model
        self.model = load_your_model(model_name)
        self.model.to(device)
    
    def _format(self, item: Query | Document) -> str:
        """Convert Query or Document to model input format."""
        # Combine text and image paths as needed
        if item.text:
            return item.text
        elif item.image:
            return item.image
        # Or combine: return f"{item.text or ''} {item.image or ''}".strip()
    
    def _compute_scores(
        self,
        query_str: str,
        doc_strs: List[str],
        query_type: str,
        doc_type: str,
        **kwargs
    ) -> List[float]:
        """Call your model API to compute scores."""
        # Create pairs and compute scores with type information
        scores = self.model.compute_similarity(
            query_str, doc_strs,
            query_type=query_type,
            doc_type=doc_type
        )
        return scores.tolist()
    
    def supported_modalities(self) -> Set[tuple]:
        """Return supported modality combinations."""
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        return {(text, image), (image, text)}

# Register the model (with optional pattern for auto-detection)
register_reranker("my-model-name", MyReranker, pattern="mymodel")

# Now you can use it
reranker = MMReranker("my-model-name")

# Local paths with "mymodel" in name will also auto-detect
reranker = MMReranker("./models/mymodel-v1")  # Auto-detected!
```

**That's it!** The base class automatically handles:
- ✅ Query type inference (`text`, `image`, `auto`)
- ✅ Document grouping by type
- ✅ Document type inference (`text`, `image`, `auto`)
- ✅ Score merging and ranking
- ✅ Modality validation

**Understanding Type Parameters:**

The `query_type` and `doc_type` parameters are automatically inferred by the base class:
- `"text"`: Pure text modality
- `"image"`: Pure image modality  
- `"auto"`: Mixed or other modalities (text + image, video, etc.)

Many multimodal models (like Jina reranker) require both `query_type` and `doc_type` to properly handle different retrieval scenarios (text-to-image, image-to-text, etc.). The base class automatically detects these types and passes them to your `_compute_scores` method.

### Adding New Metrics

Edit `evaluation/metrics.py` to add your metric function, then use it in the evaluator.

## GPU Configuration

Control GPU usage via environment variables:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 mmranker eval config.yaml

# Use CPU only
mmranker eval config.yaml  # Set device: cpu in config
```

The evaluator automatically:
- Detects available GPUs from `CUDA_VISIBLE_DEVICES`
- Distributes queries across GPUs for parallel processing
- Loads one model per GPU for efficiency

## Results

Results are saved to the output directory with two files:

### 1. Aggregated Results (`results.json`)

Summary metrics across all queries:

```json
{
  "model_name": "jinaai/jina-reranker-m0",
  "num_queries": 100,
  "num_candidates": 1000,
  "metrics": {
    "recall@1": 0.45,
    "recall@3": 0.67,
    "recall@5": 0.78,
    "recall@7": 0.84,
    "recall@10": 0.89,
    "mrr": 0.56,
    "ndcg@5": 0.68,
    "ndcg@10": 0.72
  },
  "config": {
    "recall_k": [1, 3, 5, 7, 10],
    "ndcg_k": [5, 10],
    "rank_kwargs": {"max_length": 2048}
  },
  "timestamp": "2025-10-14T10:30:00.123456"
}
```

### 2. Per-Query Results (`per_query_results.json`)

Detailed results for each query (if `save_per_query: true`):

```json
[
  {
    "query_idx": 0,
    "dataset": "VisualNews",
    "id": 2378,
    "query": {
      "text": "Sayulita beach town",
      "image": null,
      "video": null
    },
    "match": {
      "text": null,
      "image": "images/beach_town.jpg",
      "video": null
    },
    "metrics": {
      "recall@1": 1.0,
      "recall@3": 1.0,
      "recall@5": 1.0,
      "recall@7": 1.0,
      "recall@10": 1.0,
      "mrr": 1.0,
      "ndcg@5": 1.0,
      "ndcg@10": 1.0
    },
    "ranked_indices": [234, 567, 89, 123, ...],
    "scores": [0.95, 0.82, 0.76, 0.71, ...],
    "gt_idx": 234
  },
  {
    "query_idx": 1,
    "dataset": "VisualNews",
    "id": 2379,
    "query": {
      "text": null,
      "image": "images/soldier.jpg",
      "video": null
    },
    "match": {
      "text": "A South Korean soldier patrols the border",
      "image": null,
      "video": null
    },
    "metrics": {
      "recall@1": 0.0,
      "recall@3": 1.0,
      "recall@5": 1.0,
      "mrr": 0.5
    },
    "ranked_indices": [456, 123, 789, ...],
    "scores": [0.88, 0.85, 0.79, ...],
    "gt_idx": 123
  }
]
```

**Field Descriptions:**
- `query_idx`: Query index in the evaluation set
- `dataset`: Dataset name from eval data
- `id`: Sample ID from eval data
- `query`: Query content (text/image/video)
- `match`: Ground truth document
- `metrics`: Per-query metrics
- `ranked_indices`: Document indices sorted by relevance (descending)
- `scores`: Relevance scores corresponding to ranked_indices
- `gt_idx`: Index of ground truth document in candidate list

## License

MIT License

## Citation

If you use this package, please cite:

```bibtex
@software{mm_reranker_eval,
  title={MM Reranker Eval: A Unified Evaluation Package for Multimodal Rerankers},
  author={Wang, Shijian},
  year={2025},
  url={https://github.com/yourusername/mm-reranker}
}
```

