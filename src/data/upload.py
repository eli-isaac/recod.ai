"""
Upload dataset to HuggingFace Hub.
"""

from pathlib import Path

from huggingface_hub import HfApi


def upload_dataset(
    output_dir: Path,
    repo_id: str,
    private: bool = False,
) -> str:
    """
    Upload a local dataset directory to HuggingFace Hub.

    Args:
        output_dir: Local directory containing the dataset
        repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
        private: Whether the dataset should be private

    Returns:
        URL of the uploaded dataset
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    api = HfApi()

    # Create repo and upload
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    print(f"Uploading to {repo_id}...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Upload complete: {url}")
    return url
