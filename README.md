---
title: Facial Identity Persistence
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---

# Facial Identity Persistence System

Multi-image facial identity aggregation for consistent identity-preserving image generation.

## Abstract

Current identity-preserving image generation methods such as InstantID rely on a
single reference image to extract a facial embedding, which introduces instability
when the reference image has suboptimal lighting, pose, or occlusion. This project
proposes a multi-image identity aggregation system that extracts ArcFace embeddings
from 4-5 reference images and fuses them into a single robust master identity vector
using confidence-weighted averaging. The aggregated embedding is stored persistently
in a structured identity store and retrieved at generation time to condition a
Stable Diffusion XL pipeline via InstantID's IdentityNet and IP-Adapter modules.

## Problem Statement

When a single reference image is used for identity-conditioned generation:
- Poor lighting or occlusion degrades embedding quality
- A single pose biases the embedding toward that viewpoint
- There is no mechanism to refine identity representation over time

## Proposed Solution

Aggregate embeddings across multiple images of the same person:
- Extract 512-dim L2-normalized ArcFace embedding per image via InsightFace
- Weight each embedding by face detection confidence score (det_score)
- Compute weighted average and re-normalize to unit sphere
- Store master embedding as a persistent JSON record keyed by identity name
- Retrieve at generation time — no re-extraction needed

## System Architecture

```
Input: 4-5 face images
       |
InsightFace (ArcFace antelopev2)
       |
Per-image 512-dim embedding + det_score
       |
Confidence-weighted averaging + L2 normalization
       |
Master Identity Vector (512-dim)
       |
Persistent Identity Store (JSON / Neo4j roadmap)
       |
InstantID Pipeline (IdentityNet + IP-Adapter + SDXL)
       |
Output: Identity-preserved generated image
```

## Key Files

- `identity_store.py` — embedding extraction, aggregation, persistence
- `infer_multi.py` — CLI inference with saved identity support
- `gradio_demo/app_multi.py` — two-tab Gradio UI

## Novel Contributions over Base InstantID

1. Multi-image embedding aggregation vs single image
2. Confidence-weighted fusion vs equal weighting
3. Persistent identity store enabling cross-session reuse
4. Decoupled identity extraction from generation pipeline

## Experimental Direction

Planned evaluations:
- Identity similarity score (cosine similarity) between source and generated faces
- Comparison of single-image vs multi-image embedding stability across
  varying reference image quality
- Ablation: equal weighting vs confidence weighting vs PCA-based fusion

## Stack

- InstantID (Wang et al., 2024) — arXiv:2401.07519
- InsightFace / ArcFace (Deng et al., 2019)
- Stable Diffusion XL
- Gradio
- Neo4j (identity graph store, planned)

## Citation

If you build on this work:
```bibtex
@misc{instantid-multi-image,
  author = {Vikas Kumar},
  title = {Facial Identity Persistence: Multi-image Aggregation for Identity-Preserving Generation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/thisisvk45/InstantID}
}
```
