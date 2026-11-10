from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

ROOT = Path("/Users/cczh/Desktop/PEAD-Prediction")
CACHE_DIR = ROOT / "data" / "finbert_chunk_cache"
MODEL_NAME = "ProsusAI/finbert"

cache_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
cache_model = AutoModel.from_pretrained(MODEL_NAME)
cache_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
cache_model.to(cache_device)
cache_model.eval()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def tokenize_transcript(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return cache_tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_attention_mask=False, return_token_type_ids=False,
        verbose=False
    )["input_ids"]


@torch.inference_mode()
def build_chunk_cache(df_events, cache_dir=CACHE_DIR,
                      chunk_size=450, batch_size=32, storage_dtype="float16"):
    """Cache every chunk CLS, resume from the last committed batch.

    Run one writer at a time. A changed transcript/model/tokenizer/chunk
    configuration requires a new cache directory. batch_size may change.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    assert not df_events["event_id"].duplicated().any()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 1 <= chunk_size <= cache_model.config.max_position_embeddings - 2:
        raise ValueError("Invalid chunk_size for this model")
    dtype = np.dtype(storage_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("Use float16 or float32 storage")

    ids = df_events["event_id"].to_numpy(dtype=np.int64)
    texts = df_events["content"].tolist()
    source_hash = hashlib.sha256(ids.tobytes())
    for text in tqdm(texts, desc="Fingerprint transcripts"):
        normalized = text if isinstance(text, str) else ""
        source_hash.update(hashlib.sha256(normalized.encode("utf-8")).digest())

    tokenizer_hash = hashlib.sha256(json.dumps(
        cache_tokenizer.get_vocab(), sort_keys=True
    ).encode()).hexdigest()
    manifest = {
        "format_version": 1,
        "source_sha256": source_hash.hexdigest(),
        "n_events": len(ids),
        "model_name": MODEL_NAME,
        "model_revision": getattr(cache_model.config, "_commit_hash", None),
        "model_class": type(cache_model).__name__,
        "tokenizer_class": type(cache_tokenizer).__name__,
        "tokenizer_vocab_sha256": tokenizer_hash,
        "cls_token_id": cache_tokenizer.cls_token_id,
        "sep_token_id": cache_tokenizer.sep_token_id,
        "chunk_size": chunk_size,
        "stride": chunk_size,
        "embedding_method": "last_hidden_state[:, 0, :]",
        "hidden_size": cache_model.config.hidden_size,
        "storage_dtype": dtype.name,
        "transformers_version": transformers.__version__,
        "torch_version": str(torch.__version__)
    }
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Cache configuration changed. Use a new cache directory.")
    else:
        if any(cache_dir.iterdir()):
            raise ValueError("Nonempty directory without a cache manifest; use a new directory.")
        atomic_json(manifest_path, manifest)

    counts_path = cache_dir / "token_counts.npy"
    if counts_path.exists():
        token_counts = np.load(counts_path, allow_pickle=False)
    else:
        # First pass counts tokens; it does not run FinBERT.
        token_counts = np.array([
            len(tokenize_transcript(text))
            for text in tqdm(texts, desc="Count transcript tokens")
        ], dtype=np.int64)
        temporary = cache_dir / "token_counts.tmp.npy"
        np.save(temporary, token_counts)
        temporary.replace(counts_path)

    assert token_counts.shape == (len(ids),)
    chunk_counts = (token_counts + chunk_size - 1) // chunk_size
    offsets = np.concatenate(([0], np.cumsum(chunk_counts))).astype(np.int64)
    total_chunks = int(offsets[-1])
    if total_chunks == 0:
        raise ValueError("No nonempty transcripts to embed")

    np.save(cache_dir / "event_ids.npy", ids)
    np.save(cache_dir / "offsets.npy", offsets)
    np.save(cache_dir / "chunk_counts.npy", chunk_counts)
    metadata_cols = [c for c in ["event_id", "symbol", "date"] if c in df_events]
    events = df_events[metadata_cols].reset_index(drop=True).copy()
    events["token_count"] = token_counts
    events["chunk_count"] = chunk_counts
    events["chunk_start"] = offsets[:-1]
    events["chunk_stop"] = offsets[1:]
    events.to_csv(cache_dir / "events.csv", index=False)

    array_path = cache_dir / "chunk_embeddings.npy"
    progress_path = cache_dir / "progress.json"
    shape = (total_chunks, cache_model.config.hidden_size)
    written = 0
    if progress_path.exists():
        written = int(json.loads(progress_path.read_text())["written_chunks"])
    assert 0 <= written <= total_chunks
    if not array_path.exists() and written:
        raise ValueError("Progress exists but embedding array is missing")

    cache = np.lib.format.open_memmap(
        array_path,
        mode="r+" if array_path.exists() else "w+",
        dtype=dtype, shape=shape
    )
    assert cache.shape == shape and cache.dtype == dtype
    print(f"Cache: {shape}, {dtype.name}, {total_chunks * shape[1] * dtype.itemsize / 1e9:.2f} GB")

    if written == total_chunks:
        print("Cache already complete; no FinBERT inference needed.")
        return cache_dir

    cache_model.eval()
    batch_inputs = []
    batch_positions = []
    progress = tqdm(total=total_chunks, initial=written, desc="Cache FinBERT chunks")

    def process_batch():
        nonlocal written
        if not batch_inputs:
            return
        assert batch_positions == list(range(written, written + len(batch_inputs)))
        encoded = cache_tokenizer.pad(
            [{"input_ids": value} for value in batch_inputs],
            padding=True, return_tensors="pt"
        )
        encoded = {k: v.to(cache_device) for k, v in encoded.items()}
        values = cache_model(**encoded).last_hidden_state[:, 0, :].float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite embeddings; batch not committed")
        stored = values.astype(dtype)
        if not np.isfinite(stored).all():
            raise ValueError("Storage dtype overflow; batch not committed")
        end = written + len(batch_inputs)
        cache[written:end] = stored
        # Flush first, then advance the durable checkpoint.
        cache.flush()
        atomic_json(progress_path, {
            "written_chunks": end,
            "total_chunks": total_chunks,
            "complete": end == total_chunks
        })
        progress.update(end - written)
        written = end
        batch_inputs.clear()
        batch_positions.clear()

    try:
        for event_index, text in enumerate(texts):
            if offsets[event_index + 1] <= written:
                continue
            tokens = tokenize_transcript(text)
            assert len(tokens) == token_counts[event_index]
            for chunk_index, start in enumerate(range(0, len(tokens), chunk_size)):
                position = int(offsets[event_index] + chunk_index)
                if position < written:
                    continue
                chunk = tokens[start:start + chunk_size]
                batch_inputs.append(
                    [cache_tokenizer.cls_token_id] + chunk + [cache_tokenizer.sep_token_id]
                )
                batch_positions.append(position)
                if len(batch_inputs) >= batch_size:
                    process_batch()
        process_batch()
    finally:
        progress.close()
    assert written == total_chunks
    print("Complete chunk cache:", cache_dir)
    return cache_dir

