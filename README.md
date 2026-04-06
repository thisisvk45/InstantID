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

## What this does
- Upload 4-5 images of a person
- Extracts 512-dim ArcFace embeddings from each image
- Aggregates them using confidence-weighted averaging into one master identity vector
- Stores the identity persistently
- Generates new images in any style while preserving exact facial identity

## Novel contribution over InstantID
Base InstantID uses a single image. This system aggregates multiple images into a single robust identity embedding, weighted by face detection confidence, producing more stable and consistent identity preservation across generations.

## Stack
- InstantID (instantX-research)
- InsightFace / ArcFace
- Stable Diffusion XL
- Neo4j (identity graph store, roadmap)
