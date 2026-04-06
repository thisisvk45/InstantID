import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from identity_store import save_identity, load_identity, load_identity_record


# ---------------------------------------------------------------------------
# Helpers (identical to infer.py)
# ---------------------------------------------------------------------------

def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def extract_kps_from_image(image_path: str, app) -> Image.Image:
    """
    Run InsightFace on a single image and return a draw_kps PIL image
    for the largest detected face.

    Raises:
        ValueError: if no face is detected.
    """
    face_image = load_image(image_path)
    face_image = resize_img(face_image)
    face_image_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)

    faces = app.get(face_image_cv2)
    if len(faces) == 0:
        raise ValueError(f"No face detected in kps reference image: {image_path}")

    face = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    return draw_kps(face_image, face['kps'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="InstantID inference with multi-image identity aggregation."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--images",
        type=str,
        help="Comma-separated image paths to aggregate (e.g. 'img1.jpg,img2.jpg,img3.jpg')",
    )
    group.add_argument(
        "--identity",
        type=str,
        help="Name of a saved identity to load from embeddings/",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default=None,
        help="If --images is provided, save the aggregated embedding under this name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a person, best quality, high quality",
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, "
            "painting, drawing, illustration, glitch, deformed, mutated, "
            "cross-eyed, ugly, disfigured"
        ),
        help="Negative prompt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.png",
        help="Output image path (default: outputs/result.png).",
    )
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.8)
    parser.add_argument("--controlnet_scale", type=float, default=0.8)
    args = parser.parse_args()

    # Validate --save_as requirement
    if args.images and not args.save_as:
        parser.error("--save_as is required when --images is provided.")

    # ------------------------------------------------------------------
    # Step 1: Resolve master embedding + kps reference image path
    # ------------------------------------------------------------------
    if args.images:
        image_paths = [p.strip() for p in args.images.split(",") if p.strip()]
        print(f"Aggregating identity from {len(image_paths)} image(s)...")
        master_embedding = save_identity(args.save_as, image_paths)
        kps_reference_path = image_paths[0]
    else:
        print(f"Loading saved identity '{args.identity}'...")
        record = load_identity_record(args.identity)
        master_embedding = np.array(record["embedding"], dtype=np.float32)
        kps_reference_path = record["source_images"][0]
        print(f"Loaded embedding from {record['source_count']} source image(s), "
              f"created {record['created_at']}")

    # ------------------------------------------------------------------
    # Step 2: Load InsightFace (needed for kps extraction)
    # ------------------------------------------------------------------
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name='antelopev2',
        root='./',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # ------------------------------------------------------------------
    # Step 3: Extract keypoints from kps reference image
    # ------------------------------------------------------------------
    print(f"Extracting keypoints from: {kps_reference_path}")
    face_kps = extract_kps_from_image(kps_reference_path, app)

    # ------------------------------------------------------------------
    # Step 4: Load pipeline (mirrors infer.py exactly)
    # ------------------------------------------------------------------
    face_adapter = './checkpoints/ip-adapter.bin'
    controlnet_path = './checkpoints/ControlNetModel'

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    # ------------------------------------------------------------------
    # Step 5: Generate — master_embedding used exactly like face_emb in infer.py
    # ------------------------------------------------------------------
    print("Running inference...")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_embeds=master_embedding,       # (512,) — same as face_emb in infer.py:68
        image=face_kps,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    ).images[0]

    # ------------------------------------------------------------------
    # Step 6: Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()


# Example:
# python infer_multi.py --images "examples/yann-lecun_resize.jpg,examples/musk_resize.jpeg" --save_as "yann" --prompt "a man in a suit" --output outputs/yann_suit.png
# python infer_multi.py --identity "yann" --prompt "a man as an astronaut" --output outputs/yann_astro.png
