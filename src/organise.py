#!/usr/bin/env python3
"""
Organize Hugging Face model repos into collections by epoch and seed.

This script implements explicit, verbose error handling:
- Derives collection name from model repo name or CLI args
- Creates collection (logs creation errors, does NOT use exists_ok)
- Searches for existing collection by title if creation fails
- Adds model to collection
- Prints every API call, response, and exception with full diagnostics

Requires: HF_USER and HF_TOKEN environment variables
"""

import argparse
import os
import re
import sys
from typing import Optional

from huggingface_hub import HfApi


def get_hf_credentials() -> tuple[str, str]:
    """Retrieve HF_USER and HF_TOKEN from environment. Exit if missing."""
    hf_user = os.getenv("HF_USER", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not hf_user:
        print("ERROR: HF_USER environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    return hf_user, hf_token


def normalize_repo_id(repo_id: str, hf_user: str) -> str:
    """Convert repo_id to full form (namespace/name). If already namespaced, return as-is."""
    if "/" in repo_id:
        print(f"[REPO_ID] Already namespaced: {repo_id}")
        return repo_id
    else:
        full_repo_id = f"{hf_user}/{repo_id}"
        print(f"[REPO_ID] Normalized: {repo_id} -> {full_repo_id}")
        return full_repo_id


def extract_epoch_seed_from_name(repo_id: str) -> Optional[tuple[int, int]]:
    """
    Try to extract epoch and seed from repo_id using pattern -E<epoch>-S<seed>.
    Returns (epoch, seed) or None.
    """
    # Extract just the name portion if namespaced
    name_part = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    match = re.search(r"-E(\d+)-S(\d+)", name_part)
    if match:
        epoch, seed = int(match.group(1)), int(match.group(2))
        print(f"[PATTERN_MATCH] Found epoch={epoch}, seed={seed} in {name_part}")
        return epoch, seed
    print(f"[PATTERN_MATCH] No epoch/seed pattern found in {name_part}")
    return None


def determine_collection_title(repo_id: str, epoch: Optional[int], seed: Optional[int]) -> str:
    """Determine collection title from pattern or CLI args. Exit if ambiguous."""
    extracted = extract_epoch_seed_from_name(repo_id)
    if extracted:
        e, s = extracted
        title = f"MLSFT-Models-E{e}-S{s}"
        print(f"[COLLECTION_TITLE] From pattern: {title}")
        return title

    if epoch is not None and seed is not None:
        title = f"MLSFT-Models-E{epoch}-S{seed}"
        print(f"[COLLECTION_TITLE] From CLI args: {title}")
        return title

    print("ERROR: Could not determine epoch/seed. Provide in repo name (e.g. -E1-S73) or via --epoch and --seed.", file=sys.stderr)
    sys.exit(1)


def print_exception_diagnostics(exc: Exception, context: str) -> None:
    """Print full diagnostics of an exception."""
    print(f"[EXCEPTION] Context: {context}")
    print(f"[EXCEPTION] Type: {type(exc).__name__}")
    print(f"[EXCEPTION] Repr: {repr(exc)}")
    
    response = getattr(exc, "response", None)
    if response is not None:
        print(f"[EXCEPTION] HTTP status_code: {getattr(response, 'status_code', 'N/A')}")
        try:
            text = getattr(response, "text", "N/A")
            if text and len(text) > 500:
                print(f"[EXCEPTION] HTTP response (first 500 chars): {text[:500]}")
            else:
                print(f"[EXCEPTION] HTTP response: {text}")
        except Exception as e:
            print(f"[EXCEPTION] Could not read response text: {e}")
    else:
        print("[EXCEPTION] No HTTP response object attached")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize Hugging Face models into MLSFT epoch/seed collections (explicit error handling)."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Repo ID to organize (e.g., 'Qwen3-0.6B-HI-SynthDolly-E1-S73'). Can be bare name or namespace/name.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number (optional if in repo name). If provided overrides pattern match.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed number (optional if in repo name). If provided overrides pattern match.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making API calls.",
    )

    args = parser.parse_args()
    print(f"\n[START] organise.py invoked with repo_id={args.repo_id}, epoch={args.epoch}, seed={args.seed}, dry_run={args.dry_run}\n")

    # Step 1: Get credentials
    print("[STEP 1] Reading HF_USER and HF_TOKEN from environment...")
    hf_user, hf_token = get_hf_credentials()
    print(f"[STEP 1] HF_USER={hf_user}\n")

    # Step 2: Normalize repo_id
    print("[STEP 2] Normalizing repo_id...")
    full_repo_id = normalize_repo_id(args.repo_id, hf_user)
    print()

    # Step 3: Determine collection title
    print("[STEP 3] Determining collection title...")
    collection_title = determine_collection_title(full_repo_id, args.epoch, args.seed)
    print()

    if args.dry_run:
        print("[DRY_RUN] Would create collection:", collection_title)
        print("[DRY_RUN] Would add model:", full_repo_id)
        print("[DRY_RUN] Exiting (dry run mode).")
        sys.exit(0)

    # Step 4: Create HfApi client
    print("[STEP 4] Creating HfApi client...")
    try:
        api = HfApi(token=hf_token)
        print("[STEP 4] HfApi client created successfully.\n")
    except Exception as e:
        print_exception_diagnostics(e, "HfApi initialization")
        print("ERROR: Could not create HfApi client.", file=sys.stderr)
        sys.exit(1)

    # Step 5: Verify model exists
    print("[STEP 5] Checking if model exists on Hugging Face...")
    print(f"[API] Calling api.model_info(repo_id={full_repo_id!r}, token=***)")
    try:
        model_info = api.model_info(repo_id=full_repo_id, token=hf_token)
        print(f"[API_RESPONSE] Model found: id={model_info.id}, private={model_info.private}, tags={model_info.tags}\n")
    except Exception as e:
        print_exception_diagnostics(e, "model_info")
        print(f"WARNING: Model {full_repo_id} not found or not accessible. Exiting gracefully.", file=sys.stderr)
        sys.exit(0)

    # Step 6: Attempt to create collection
    print("[STEP 6] Attempting to create collection (exists_ok=False, will error if exists)...")
    print(f"[API] Calling api.create_collection(")
    print(f"       title={collection_title!r},")
    print(f"       namespace={hf_user!r},")
    print(f"       description='Fine-tuned models from the MLSFT pipeline: {collection_title}',")
    print(f"       private=False,")
    print(f"       exists_ok=False,")
    print(f"       token=***)")
    
    collection_slug = None
    creation_error = None

    try:
        collection = api.create_collection(
            title=collection_title,
            namespace=hf_user,
            description=f"Fine-tuned models from the MLSFT pipeline: {collection_title}",
            private=False,
            exists_ok=False,
            token=hf_token,
        )
        collection_slug = collection.slug
        print(f"[API_RESPONSE] Collection created successfully: slug={collection_slug}\n")
    except Exception as e:
        creation_error = e
        print_exception_diagnostics(e, "create_collection")
        print("[FALLBACK] Collection creation failed; will attempt to find existing collection by title.\n")

    # Step 7: If creation failed, search for existing collection
    if collection_slug is None:
        print("[STEP 7] Searching for existing collection by title...")
        print(f"[API] Calling api.list_collections(owner={hf_user!r}, token=***)")
        
        try:
            collections_iter = api.list_collections(owner=hf_user, token=hf_token)
            collections = list(collections_iter)
            print(f"[API_RESPONSE] Found {len(collections)} collection(s) owned by {hf_user}")
            
            for coll in collections:
                print(f"  - title={coll.title!r}, slug={coll.slug!r}")
                if coll.title == collection_title:
                    collection_slug = coll.slug
                    print(f"[MATCH] Collection matched by title: {collection_slug}\n")
                    break
            
            if collection_slug is None:
                print(f"[FALLBACK] No collection with title {collection_title!r} found in user's collections.", file=sys.stderr)
                print("[FALLBACK] Attempting direct get_collection as last resort...\n")
                
                # Last resort: try to get collection by constructed slug
                try:
                    guessed_slug = f"{hf_user}/{collection_title}"
                    print(f"[API] Calling api.get_collection(collection_slug={guessed_slug!r}, token=***)")
                    coll = api.get_collection(collection_slug=guessed_slug, token=hf_token)
                    collection_slug = coll.slug
                    print(f"[API_RESPONSE] Collection found by guessed slug: {collection_slug}\n")
                except Exception as e2:
                    print_exception_diagnostics(e2, "get_collection (guessed slug)")
                    print("ERROR: Could not locate existing collection. Aborting.", file=sys.stderr)
                    sys.exit(1)
        except Exception as e:
            print_exception_diagnostics(e, "list_collections")
            print("ERROR: Could not list collections. Aborting.", file=sys.stderr)
            sys.exit(1)

    # Step 8: Add model to collection
    print("[STEP 8] Adding model to collection...")
    print(f"[API] Calling api.add_collection_item(")
    print(f"       collection_slug={collection_slug!r},")
    print(f"       item_id={full_repo_id!r},")
    print(f"       item_type='model',")
    print(f"       exists_ok=False,")
    print(f"       token=***)")
    
    try:
        result = api.add_collection_item(
            collection_slug=collection_slug,
            item_id=full_repo_id,
            item_type="model",
            exists_ok=False,
            token=hf_token,
        )
        print(f"[API_RESPONSE] Item added successfully. Collection now has {len(result.items)} items.\n")
    except Exception as e:
        print_exception_diagnostics(e, "add_collection_item")
        print("ERROR: Could not add model to collection. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 9: Verify by fetching collection
    print("[STEP 9] Verifying: fetching collection to confirm model was added...")
    print(f"[API] Calling api.get_collection(collection_slug={collection_slug!r}, token=***)")
    
    try:
        final_collection = api.get_collection(collection_slug=collection_slug, token=hf_token)
        print(f"[API_RESPONSE] Collection: title={final_collection.title!r}, items={len(final_collection.items)}")
        
        # Print last few items
        if final_collection.items:
            print("  Last few items in collection:")
            for item in final_collection.items[-3:]:
                print(f"    - {item.item_id} ({item.item_type})")
        print()
    except Exception as e:
        print_exception_diagnostics(e, "get_collection (verification)")
        print("WARNING: Could not verify collection state, but add_collection_item succeeded.", file=sys.stderr)

    print("[SUCCESS] Model organized into collection.")
    print(f"  Collection slug: {collection_slug}")
    print(f"  Model repo: {full_repo_id}")
    print(f"  Collection title: {collection_title}\n")


if __name__ == "__main__":
    main()
