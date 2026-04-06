import os
import json
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from insightface.app import FaceAnalysis

# Initialized lazily to avoid loading the model on import
_app = None

def _get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(
            name='antelopev2',
            root='./',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def extract_embedding(image_path: str) -> np.ndarray:
    """
    Load image, detect faces with InsightFace, and return the 512-dim embedding
    for the largest detected face.

    Returns:
        np.ndarray of shape (512,)

    Raises:
        ValueError: if no face is detected in the image.
    """
    image = Image.open(image_path).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    app = _get_app()
    faces = app.get(image_cv2)

    if len(faces) == 0:
        raise ValueError(f"No face detected in image: {image_path}")

    # Pick largest face by bounding box area
    face = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    return face['embedding']  # shape (512,), float32


def aggregate_embeddings(image_paths: list) -> np.ndarray:
    """
    Extract face embeddings from each image, compute a confidence-weighted average,
    and return an L2-normalized master embedding.

    Args:
        image_paths: list of 1 or more image file paths.

    Returns:
        np.ndarray of shape (512,), L2-normalized.

    Raises:
        ValueError: if any image yields no face, or if image_paths is empty.
    """
    if not image_paths:
        raise ValueError("image_paths must contain at least one path.")

    app = _get_app()

    embeddings = []
    weights = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces = app.get(image_cv2)
        if len(faces) == 0:
            raise ValueError(f"No face detected in image: {path}")

        face = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]

        embeddings.append(face['embedding'])
        # det_score is InsightFace's detection confidence [0, 1]
        weights.append(float(face.get('det_score', 1.0)))

    embeddings = np.stack(embeddings, axis=0)   # (N, 512)
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()                     # normalize weights to sum to 1

    aggregated = (embeddings * weights[:, None]).sum(axis=0)  # (512,)

    # L2-normalize
    norm = np.linalg.norm(aggregated)
    if norm > 0:
        aggregated = aggregated / norm

    return aggregated.astype(np.float32)


def save_identity(name: str, image_paths: list, save_dir: str = "./embeddings") -> np.ndarray:
    """
    Aggregate embeddings from image_paths and save to {save_dir}/{name}.json.

    JSON structure:
        {
            "name": str,
            "embedding": [512 floats],
            "source_count": int,
            "created_at": ISO-8601 timestamp
        }

    Returns:
        np.ndarray of shape (512,) — the saved master embedding.
    """
    os.makedirs(save_dir, exist_ok=True)

    embedding = aggregate_embeddings(image_paths)

    record = {
        "name": name,
        "embedding": embedding.tolist(),
        "source_images": [str(p) for p in image_paths],
        "source_count": len(image_paths),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    save_path = os.path.join(save_dir, f"{name}.json")
    with open(save_path, "w") as f:
        json.dump(record, f)

    print(f"Saved identity '{name}' ({len(image_paths)} source images) → {save_path}")
    return embedding


def load_identity(name: str, save_dir: str = "./embeddings") -> np.ndarray:
    """
    Load a saved identity embedding from {save_dir}/{name}.json.

    Returns:
        np.ndarray of shape (512,)

    Raises:
        FileNotFoundError: if the identity file does not exist.
    """
    load_path = os.path.join(save_dir, f"{name}.json")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No saved identity found at: {load_path}")

    with open(load_path, "r") as f:
        record = json.load(f)

    return np.array(record["embedding"], dtype=np.float32)


def load_identity_record(name: str, save_dir: str = "./embeddings") -> dict:
    """
    Load the full JSON record for a saved identity.

    Returns:
        dict with keys: name, embedding, source_images, source_count, created_at

    Raises:
        FileNotFoundError: if the identity file does not exist.
    """
    load_path = os.path.join(save_dir, f"{name}.json")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No saved identity found at: {load_path}")

    with open(load_path, "r") as f:
        return json.load(f)


def list_identities(save_dir: str = "./embeddings") -> list:
    """
    Return a list of all saved identity names in save_dir.
    """
    if not os.path.exists(save_dir):
        return []
    return [
        os.path.splitext(fname)[0]
        for fname in os.listdir(save_dir)
        if fname.endswith(".json")
    ]


if __name__ == "__main__":
    image1 = "./examples/yann-lecun_resize.jpg"
    image2 = "./examples/musk_resize.jpeg"

    print("=== save_identity ===")
    embedding = save_identity("test_person", [image1, image2])

    print("\n=== load_identity ===")
    loaded = load_identity("test_person")

    print(f"Shape : {loaded.shape}")
    print(f"First 5 values: {loaded[:5]}")

    print("\n=== list_identities ===")
    print(list_identities())
