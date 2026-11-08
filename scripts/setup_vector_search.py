"""One-time setup: create a Vector Search index, a public endpoint, and deploy.

Usage:
  python scripts/setup_vector_search.py

Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION / EMBEDDING_DIMENSIONS from
rag_agent/.env and writes the created resource names back into that file.
Deployment usually takes 20-40 minutes.
"""

import os
import pathlib
import re

from dotenv import load_dotenv
from google.cloud import aiplatform

ENV_FILE = pathlib.Path(__file__).resolve().parents[1] / "rag_agent" / ".env"
DEPLOYED_INDEX_ID = "rag_agent_deployed_index"


def set_env_value(key: str, value: str) -> None:
  text = ENV_FILE.read_text()
  line = f"{key}={value}"
  if re.search(rf"^{key}=.*$", text, flags=re.M):
    text = re.sub(rf"^{key}=.*$", line, text, flags=re.M)
  else:
    text += f"\n{line}\n"
  ENV_FILE.write_text(text)


def main() -> None:
  load_dotenv(ENV_FILE)
  aiplatform.init(
      project=os.environ["GOOGLE_CLOUD_PROJECT"],
      location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
  )

  print("Creating streaming index (a few minutes)...")
  index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
      display_name="rag-agent-index",
      dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
      approximate_neighbors_count=150,
      distance_measure_type="DOT_PRODUCT_DISTANCE",
      feature_norm_type="UNIT_L2_NORM",
      index_update_method="STREAM_UPDATE",
      shard_size="SHARD_SIZE_SMALL",
  )
  set_env_value("VECTOR_SEARCH_INDEX", index.resource_name)
  print(f"Index: {index.resource_name}")

  print("Creating public index endpoint...")
  endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
      display_name="rag-agent-endpoint", public_endpoint_enabled=True
  )
  set_env_value("VECTOR_SEARCH_INDEX_ENDPOINT", endpoint.resource_name)
  print(f"Endpoint: {endpoint.resource_name}")

  print("Deploying index to endpoint (typically 20-40 minutes)...")
  endpoint.deploy_index(
      index=index,
      deployed_index_id=DEPLOYED_INDEX_ID,
      machine_type="e2-standard-2",
      min_replica_count=1,
      max_replica_count=1,
  )
  set_env_value("VECTOR_SEARCH_DEPLOYED_INDEX_ID", DEPLOYED_INDEX_ID)
  print(f"Done. Values written to {ENV_FILE}")


if __name__ == "__main__":
  main()
