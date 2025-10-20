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
uv pip install -e .

# With flash attention (recommended for better performance)
uv pip install -e '.[flash-attention]'

# With development tools
uv pip install -e '.[dev]'
```

### 5. Deactivate environment

```bash
deactivate
```

### 6. Switch to different environment

```bash
# Deactivate current environment
deactivate

# Activate another environment
source scripts/activate_env.sh /path/to/another/env
```

## Quick Start

### 1. Programmatic Usage

```python
from mm_reranker_eval import MMReranker, Query, Document

# Initialize reranker with remote model name
reranker = MMReranker("jinaai/jina-reranker-m0", device="cuda")

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
result = reranker.rank(query, documents)
print(f"Ranked indices: {result.ranked_indices}")
print(f"Scores: {result.scores}")
```

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

```python
from mm_reranker_eval.reranker import BaseReranker, register_reranker
from mm_reranker_eval.data.types import Query, Document, RankResult, Modality

class MyReranker(BaseReranker):
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        # Load your model
        
    def rank(self, query: Query, documents: List[Document], **kwargs) -> RankResult:
        # Implement ranking logic
        pass
        
    def supported_modalities(self) -> Set[tuple]:
        # Return supported modality combinations
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

Results are saved to the output directory:

**results.json:**
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
    "mrr": 0.56
  },
  "config": {...},
  "timestamp": "2025-10-14T10:30:00"
}
```

**per_query_results.json** (if `save_per_query: true`):
```json
[
  {
    "query_idx": 0,
    "dataset": "VisualNews",
    "id": 2378,
    "metrics": {"recall@1": 1.0, "recall@3": 1.0, ...}
  },
  ...
]
```

## Design Principles

1. **Simplicity**: Unified interface, minimal APIs
2. **Extensibility**: Easy to add new models, metrics, and modalities
3. **Type Safety**: Full Python typing and docstrings
4. **Performance**: Parallel GPU evaluation, efficient processing
5. **Clarity**: Clean naming (mm vs multimodal), clear code structure

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

