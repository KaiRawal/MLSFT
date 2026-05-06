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
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


def get_hf_credentials() -> tuple[str, str]:
    """Retrieve HF_USER and HF_TOKEN from environment."""
    hf_user = os.getenv("HF_USER")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_user:
        raise ValueError("HF_USER environment variable is not set.")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    return hf_user, hf_token


def get_repo_info(api: HfApi, repo_id: str) -> dict | None:
    """Get info about a repo. Returns None if not found."""
    try:
        return api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return None


def collection_exists(api: HfApi, hf_user: str, collection_slug: str) -> bool:
    """Check if a collection exists for the user."""
    try:
        api.get_collection(collection_slug=f"{hf_user}/{collection_slug}")
        return True
    except RepositoryNotFoundError:
        return False


def get_or_create_collection(
    api: HfApi, hf_user: str, collection_slug: str, token: str
) -> dict:
    """
    Get an existing collection or create a new one.

    Returns the collection object.
    """
    if collection_exists(api, hf_user, collection_slug):
        return api.get_collection(collection_slug=f"{hf_user}/{collection_slug}")

    # Collection doesn't exist, create it
    print(f"Creating new collection: {collection_slug}")
    collection = api.create_collection(
        collection_name=collection_slug,
        description=f"Fine-tuned models from the MLSFT pipeline: {collection_slug}",
        private=False,
        token=token,
    )
    return collection


def repo_in_any_collection(api: HfApi, repo_id: str) -> bool:
    """
    Check if a repo is already in any collection.

    Uses the repo info to detect collection membership.
    """
    repo_info = get_repo_info(api, repo_id)
    if repo_info is None:
        return False

    # Check if the repo has any collections (the collections attribute)
    # Note: The exact attribute name may vary; we check for presence of collection refs
    if hasattr(repo_info, "collections") and repo_info.collections:
        return True

    return False


def add_repo_to_collection(
    api: HfApi,
    collection_slug: str,
    repo_id: str,
    token: str,
) -> None:
    """Add a repo to a collection."""
    full_collection_slug = f"{collection_slug}"
    api.add_collection_item(
        collection_slug=full_collection_slug,
        item_id=repo_id,
        item_type="model",
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

    # Construct the collection slug
    collection_slug = f"MLSFT-Models-E{args.epoch}-S{args.seed}"

    print(f"Organizing repo {full_repo_id} into collection {collection_slug}...")

    # Check if the repo is already in any collection
    if repo_in_any_collection(api, full_repo_id):
        print(f"Repo {full_repo_id} is already in a collection. Skipping.")
        sys.exit(0)

    # Get or create the collection
    try:
        collection = get_or_create_collection(api, hf_user, collection_slug, hf_token)
        print(f"Collection {collection_slug} ready.")
    except Exception as e:
        print(f"Error managing collection: {e}", file=sys.stderr)
        sys.exit(1)

    # Add the repo to the collection
    try:
        add_repo_to_collection(
            api,
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
