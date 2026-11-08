"""Chunk, embed and upsert documents into the Vector Search index.

Usage:
  python scripts/ingest.py [DIR_OR_FILE ...]   # default: data/sample

Supports .txt and .md files. Chunk text and source path are stored in each
datapoint's embedding_metadata, so the agent needs no separate document store.
"""

import hashlib
import os
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "rag_agent" / ".env")

from google.cloud import aiplatform  # noqa: E402
from google.cloud.aiplatform_v1.types import IndexDatapoint  # noqa: E402

from rag_agent.vector_search import embed_texts  # noqa: E402

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 150
EMBED_BATCH = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
  """Splits text into ~size-char chunks on paragraph boundaries when possible."""
  paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
  chunks, current = [], ""
  for p in paragraphs:
    while len(p) > size:  # hard-split very long paragraphs
      if current:
        chunks.append(current)
        current = ""
      chunks.append(p[:size])
      p = p[size - overlap:]
    if current and len(current) + len(p) + 2 > size:
      chunks.append(current)
      current = current[-overlap:] + "\n\n" + p
    else:
      current = f"{current}\n\n{p}" if current else p
  if current:
    chunks.append(current)
  return chunks


def iter_files(paths):
  for raw in paths:
    path = pathlib.Path(raw)
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for f in files:
      if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
        yield f


def build_datapoints(files):
  records = []
  for f in files:
    source = str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f)
    for i, chunk in enumerate(chunk_text(f.read_text(encoding="utf-8"))):
      dp_id = hashlib.sha1(f"{source}#{i}".encode()).hexdigest()
      records.append((dp_id, chunk, source))
  datapoints = []
  for start in range(0, len(records), EMBED_BATCH):
    batch = records[start:start + EMBED_BATCH]
    vectors = embed_texts([r[1] for r in batch], task_type="RETRIEVAL_DOCUMENT")
    for (dp_id, chunk, source), vec in zip(batch, vectors):
      datapoints.append(
          IndexDatapoint(
              datapoint_id=dp_id,
              feature_vector=vec,
              embedding_metadata={"text": chunk, "source": source},
          )
      )
  return datapoints


def main(argv):
  paths = argv or [str(ROOT / "data" / "sample")]
  files = list(iter_files(paths))
  if not files:
    sys.exit(f"No .txt/.md files found in {paths}")
  datapoints = build_datapoints(files)
  print(f"Embedded {len(datapoints)} chunks from {len(files)} files.")

  aiplatform.init(
      project=os.environ["GOOGLE_CLOUD_PROJECT"],
      location=os.getenv("VECTOR_SEARCH_LOCATION") or "us-central1",
  )
  index = aiplatform.MatchingEngineIndex(
      index_name=os.environ["VECTOR_SEARCH_INDEX"]
  )
  for start in range(0, len(datapoints), 100):
    index.upsert_datapoints(datapoints=datapoints[start:start + 100])
  print("Upserted. Streaming updates are searchable within a few seconds.")


if __name__ == "__main__":
  main(sys.argv[1:])
