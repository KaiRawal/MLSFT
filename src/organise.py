#!/usr/bin/env python3
"""
Organize Hugging Face model repos into collections by epoch and seed.

This script manages collections for fine-tuned models produced by the MLSFT pipeline.
Each collection groups models from a specific epoch/seed pair:
  - Collection name format: MLSFT-Models-E{epoch}-S{seed}
  - Example: MLSFT-Models-E1-S73

The script is idempotent:
  - Reuses existing collections if they already exist
  - Skips repos that are already in any collection
  - Only adds uncollected repos to the target collection
"""

import argparse
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def get_hf_credentials() -> tuple[str, str]:
    """Retrieve HF_USER and HF_TOKEN from environment."""
    hf_user = os.getenv("HF_USER")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_user:
        raise ValueError("HF_USER environment variable is not set.")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    return hf_user, hf_token


def get_repo_info(api: HfApi, repo_id: str, token: str | None = None) -> dict | None:
    """Get info about a repo. Returns None if not found.

    Uses `model_info` and treats HTTP 404 as "not found". Other HTTP
    errors are re-raised for the caller to handle.
    """
    try:
        return api.model_info(repo_id=repo_id, token=token)
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            return None
        raise


def get_or_create_collection(api: HfApi, hf_user: str, collection_title: str, token: str):
    """Get an existing collection or create a new one (atomic).

    The HF client supports `create_collection(..., exists_ok=True)` which
    returns the existing collection if it already exists. Use that to
    avoid race conditions and to simplify error handling.
    """
    slug = f"{hf_user}/{collection_title}"
    try:
        collection = api.create_collection(
            title=collection_title,
            namespace=hf_user,
            description=f"Fine-tuned models from the MLSFT pipeline: {collection_title}",
            private=False,
            exists_ok=True,
            token=token,
        )
        return collection
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 403:
            raise RuntimeError(
                f"Permission denied interacting with collection '{slug}'. Verify HF_TOKEN permissions and namespace."
            ) from exc
        raise


def repo_in_any_collection(api: HfApi, hf_user: str, repo_id: str, token: str) -> bool:
    """Check whether the repo is already present in any collection owned by the user.

    The HF API accepts item filters in the form `models/<namespace>/<repo>` or
    `models/<repo>`. Try both variants when given a namespaced repo id.
    If listing collections fails, return False so the caller can attempt
    to add the repo (the add call will surface errors if any).
    """
    candidates: list[str] = []
    if "/" in repo_id:
        namespace, name = repo_id.split("/", 1)
        candidates.append(f"models/{namespace}/{name}")
        candidates.append(f"models/{name}")
    else:
        candidates.append(f"models/{repo_id}")

    try:
        for item in candidates:
            collections = api.list_collections(owner=hf_user, item=item, token=token)
            if collections:
                return True
    except HfHubHTTPError as exc:
        print(f"Warning: unable to list collections for owner {hf_user}: {exc}", file=sys.stderr)
        return False

    return False


def add_repo_to_collection(
    api: HfApi,
    hf_user: str,
    collection_slug: str,
    repo_id: str,
    token: str,
) -> None:
    """Add a repo to a collection."""
    api.add_collection_item(
        collection_slug=collection_slug,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize Hugging Face models into MLSFT epoch/seed collections."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Repo ID to organize (e.g., 'Gemma-3-1B-IT-HI-SynthDolly-1A-E1-S73')",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Number of training epochs for collection naming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for collection naming",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.epoch <= 0:
        print("Error: epoch must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if args.seed <= 0:
        print("Error: seed must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    try:
        hf_user, hf_token = get_hf_credentials()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=hf_token)

    # Construct the full repo ID (user/repo_id)
    full_repo_id = f"{hf_user}/{args.repo_id}"

    # Check if the repo exists
    repo_info = get_repo_info(api, full_repo_id)
    if repo_info is None:
        print(f"Warning: Repo {full_repo_id} not found. It may not have been uploaded yet.", file=sys.stderr)
        sys.exit(0)

    # Construct the collection title; the HF API will return the real slug.
    collection_title = f"MLSFT-Models-E{args.epoch}-S{args.seed}"

    print(f"Organizing repo {full_repo_id} into collection {collection_title}...")

    # Check if the repo is already in any collection
    if repo_in_any_collection(api, hf_user, full_repo_id, hf_token):
        print(f"Repo {full_repo_id} is already in a collection. Skipping.")
        sys.exit(0)

    # Get or create the collection
    try:
        collection = get_or_create_collection(api, hf_user, collection_title, hf_token)
        collection_slug = collection.slug
        print(f"Collection {collection_slug} ready.")
    except Exception as e:
        print(f"Error managing collection: {e}", file=sys.stderr)
        sys.exit(1)

    # Add the repo to the collection
    try:
        add_repo_to_collection(
            api,
            hf_user,
            collection_slug,
            full_repo_id,
            hf_token,
        )
        print(f"Successfully added {full_repo_id} to collection {collection_slug}.")
    except Exception as e:
        print(f"Error adding repo to collection: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
