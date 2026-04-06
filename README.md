---
title: Facial Identity Persistence
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: true
---

# Facial Identity Persistence System

Multi-image facial identity aggregation for consistent identity-preserving image generation.
Built on top of InstantID (Wang et al., 2024) with novel contributions in embedding fusion and persistent identity storage.

## The Problem

State-of-the-art identity-preserving generation models like InstantID condition on a single reference image. This creates three failure modes:

- A single image with poor lighting, occlusion, or extreme pose produces a degraded embedding
- The embedding is biased toward one viewpoint and expression
- There is no way to persist or reuse an identity across sessions without re-running extraction

## What This System Does Differently

This project introduces a multi-image identity aggregation layer on top of InstantID:

- Accepts 4-5 reference images of the same person
- Extracts a 512-dim ArcFace embedding from each image using InsightFace
- Fuses embeddings using confidence-weighted averaging weighted by face detection confidence score
- Re-normalizes the fused vector to the unit sphere
- Persists the master identity vector in a structured store keyed by name
- Retrieves the stored identity at generation time with no re-extraction needed

## Architecture

```
4-5 Reference Images
        |
InsightFace ArcFace antelopev2
        |
Per-image 512-dim embedding + detection confidence score
        |
Confidence-weighted average + L2 normalization
        |
Master Identity Vector 512-dim
        |
Persistent Identity Store JSON
        |
InstantID Pipeline
  |- IdentityNet spatial conditioning via facial keypoints
  |- IP-Adapter semantic conditioning via face tokens
  |- Stable Diffusion XL
        |
Identity-preserved output image
```

## Novel Contributions Over Base InstantID

| Aspect | Base InstantID | This System |
|---|---|---|
| Reference images | 1 | 4-5 |
| Embedding fusion | None | Confidence-weighted average |
| Identity persistence | None | JSON store keyed by name |
| Cross-session reuse | Not supported | Supported |
| Pipeline modification | Required | Zero, drop-in replacement |

## Key Files

| File | Description |
|---|---|
| identity_store.py | Embedding extraction, confidence-weighted aggregation, persistent storage |
| infer_multi.py | CLI inference supporting multi-image input and saved identity retrieval |
| gradio_demo/app_multi.py | Two-tab Gradio UI for identity creation and generation |

## Usage

Save a new identity:
```
python infer_multi.py --images "img1.jpg,img2.jpg,img3.jpg" --save_as "john" --prompt "a man in a suit" --output outputs/john_suit.png
```

Generate using a saved identity:
```
python infer_multi.py --identity "john" --prompt "a man as an astronaut" --output outputs/john_astro.png
```

## Planned Evaluations

- Cosine similarity between source and generated face embeddings across single vs multi-image baselines
- Embedding stability under varying reference image quality
- Ablation: equal weighting vs confidence weighting vs PCA-based fusion

## References

- InstantID: Wang et al., 2024. arXiv:2401.07519
- ArcFace: Deng et al., 2019. arXiv:1801.07698
- Stable Diffusion XL: Podell et al., 2023. arXiv:2307.01952
- InsightFace: github.com/deepinsight/insightface

## Citation

```
@misc{facial-identity-persistence,
  author = {Vikas Kumar},
  title = {Facial Identity Persistence: Multi-image Aggregation for Identity-Preserving Generation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/thisisvk45/InstantID}
}
```
