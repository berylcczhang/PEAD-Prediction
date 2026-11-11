from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm

ROOT = Path("/Users/cczh/Desktop/PEAD-Prediction")
SOURCE_CACHE = ROOT / "data" / "finbert_chunk_cache"
SENTIMENT_CACHE = ROOT / "data" / "finbert_chunk_sentiment_cache"
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

source_manifest = json.loads((SOURCE_CACHE / "manifest.json").read_text())
source_progress = json.loads((SOURCE_CACHE / "progress.json").read_text())
assert source_progress["complete"], "Complete the embedding cache first."
assert source_manifest["embedding_method"] == "last_hidden_state[:, 0, :]"
assert source_manifest["model_class"] == "BertModel"
revision = source_manifest.get("model_revision")
if not revision:
    raise ValueError("Source cache must record an exact model revision.")

sentiment_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# Load the SAME checkpoint revision as the cached CLS vectors.
# The full encoder is loaded on CPU, but is not run.
classification_model = AutoModelForSequenceClassification.from_pretrained(
    source_manifest["model_name"], revision=revision
).eval()
assert classification_model.config.model_type == "bert"
assert classification_model.bert.pooler is not None
id2label = {
    int(i): str(label).strip().lower()
    for i, label in classification_model.config.id2label.items()
}
label_to_id = {label: i for i, label in id2label.items()}
assert set(label_to_id) == set(SENTIMENT_LABELS)
column_order = [label_to_id[label] for label in SENTIMENT_LABELS]

# Keep only the pretrained pooler and classification head in memory.
sentiment_pooler = classification_model.bert.pooler.to(sentiment_device).eval()
sentiment_dropout = classification_model.dropout.to(sentiment_device).eval()
sentiment_classifier = classification_model.classifier.to(sentiment_device).eval()
del classification_model
print("Score column order:", SENTIMENT_LABELS)
print("Device:", sentiment_device)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


@torch.inference_mode()
def cache_chunk_sentiment(source_dir=SOURCE_CACHE,
                          output_dir=SENTIMENT_CACHE, batch_size=4096):
    """One writer at a time. Score cached CLS; resume after committed batches."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if source_dir.resolve() != SOURCE_CACHE.resolve():
        raise ValueError("Reload the matching sentiment head for a different source cache.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    status = json.loads((source_dir / "progress.json").read_text())
    assert status["complete"]

    embeddings = np.load(source_dir / "chunk_embeddings.npy", mmap_mode="r")
    ids = np.load(source_dir / "event_ids.npy", allow_pickle=False)
    offsets = np.load(source_dir / "offsets.npy", allow_pickle=False)
    assert len(offsets) == len(ids) + 1
    assert offsets[0] == 0 and np.all(np.diff(offsets) >= 0)
    assert len(np.unique(ids)) == len(ids)
    n_chunks = len(embeddings)
    assert offsets[-1] == n_chunks == status["written_chunks"] == status["total_chunks"]
    assert embeddings.shape[1] == source_manifest["hidden_size"]
    assert source_manifest["n_events"] == len(ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "source_manifest_sha256": hashlib.sha256(
            (source_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_model_revision": revision,
        "source_embedding_dtype": str(embeddings.dtype),
        "method": "cached CLS -> pretrained BERT pooler -> dropout(eval) -> classifier -> softmax",
        "columns": SENTIMENT_LABELS,
        "storage_dtype": "float32",
        "n_chunks": n_chunks,
        "n_events": len(ids),
        "event_aggregation": "equal-weight mean of chunk probabilities"
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing sentiment cache belongs to different inputs/configuration.")
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Use an empty output directory.")
        atomic_json(manifest_path, manifest)

    for filename in ["event_ids.npy", "offsets.npy", "chunk_counts.npy", "token_counts.npy", "events.csv"]:
        shutil.copyfile(source_dir / filename, output_dir / filename)

    progress_path = output_dir / "progress.json"
    written = int(json.loads(progress_path.read_text())["written_chunks"]) if progress_path.exists() else 0
    assert 0 <= written <= n_chunks
    array_path = output_dir / "chunk_sentiment.npy"
    if written and not array_path.exists():
        raise ValueError("Sentiment array missing despite saved progress.")
    scores = np.lib.format.open_memmap(
        array_path, mode="r+" if array_path.exists() else "w+",
        dtype=np.float32, shape=(n_chunks, 3)
    )
    assert scores.shape == (n_chunks, 3) and scores.dtype == np.dtype("float32")
    sentiment_pooler.eval()
    sentiment_dropout.eval()
    sentiment_classifier.eval()

    progress = tqdm(total=n_chunks, initial=written,
                    desc="Cache chunk sentiment", unit="chunk")
    try:
        for start in range(written, n_chunks, batch_size):
            end = min(start + batch_size, n_chunks)
            cls = torch.from_numpy(np.array(
                embeddings[start:end], dtype=np.float32, copy=True
            )).to(sentiment_device)
            # BERT's pooler takes [batch, tokens, hidden]; only CLS is needed.
            pooled = sentiment_pooler(cls.unsqueeze(1))
            logits = sentiment_classifier(sentiment_dropout(pooled))
            probability = torch.softmax(logits.float(), dim=-1)[:, column_order].cpu().numpy()
            assert np.isfinite(probability).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
            scores[start:end] = probability
            scores.flush()
            atomic_json(progress_path, {
                "written_chunks": end, "total_chunks": n_chunks,
                "complete": end == n_chunks
            })
            progress.update(end - start)
    finally:
        progress.close()

    # Also provide transcript-level arrays in the same event order.
    event_scores = np.full((len(ids), 3), np.nan, dtype=np.float32)
    for i in range(len(ids)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if end > start:
            event_scores[i] = scores[start:end].mean(axis=0, dtype=np.float64)
    np.save(output_dir / "event_sentiment.npy", event_scores)
    events = pd.read_csv(output_dir / "events.csv")
    np.testing.assert_array_equal(events["event_id"], ids)
    for j, label in enumerate(SENTIMENT_LABELS):
        events[f"sentiment_{label}"] = event_scores[:, j]
    events["sentiment_score"] = event_scores[:, 0] - event_scores[:, 2]
    events.to_csv(output_dir / "event_sentiment.csv", index=False)
    print("Chunk sentiment:", scores.shape)
    print("Event sentiment:", event_scores.shape)
    print("Saved:", output_dir)
    return output_dir

