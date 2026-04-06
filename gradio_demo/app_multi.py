import os
import sys

# Add repo root so we can import pipeline files and identity_store
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import math
import numpy as np
import torch
import diffusers
from PIL import Image

import gradio as gr
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from model_util import get_torch_device
from style_template import styles
from identity_store import save_identity, load_identity, load_identity_record, list_identities

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "embeddings")
FACE_ADAPTER = os.path.join(REPO_ROOT, "checkpoints", "ip-adapter.bin")
CONTROLNET_PATH = os.path.join(REPO_ROOT, "checkpoints", "ControlNetModel")
BASE_MODEL = "wangqixun/YamerMIX_v8"

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE = "(No style)"

device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# ---------------------------------------------------------------------------
# Startup: load models once
# ---------------------------------------------------------------------------
print("Loading FaceAnalysis...")
face_app = FaceAnalysis(
    name="antelopev2",
    root=REPO_ROOT,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=dtype)

print(f"Loading base pipeline ({BASE_MODEL})...")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
).to(device)
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter_instantid(FACE_ADAPTER)
print("Pipeline ready.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resize_img(input_image, max_side=1280, min_side=1024,
               pad_to_max_side=False, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], Image.BILINEAR)
    w_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_new, h_new], Image.BILINEAR)
    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        ox, oy = (max_side - w_new) // 2, (max_side - h_new) // 2
        res[oy:oy + h_new, ox:ox + w_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def extract_kps(image_path: str) -> tuple[Image.Image, int, int]:
    """
    Run FaceAnalysis on image_path and return (kps_image, height, width).
    Picks the largest detected face.
    Raises ValueError if no face found.
    """
    img = load_image(image_path)
    img = resize_img(img)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    faces = face_app.get(img_cv2)
    if not faces:
        raise ValueError(f"No face detected in: {image_path}")

    face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
    kps_img = draw_kps(img, face["kps"])
    h, w = img_cv2.shape[:2]
    return kps_img, h, w


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE])
    return p.replace("{prompt}", positive), n + " " + negative


# ---------------------------------------------------------------------------
# Tab 1 — Create Identity
# ---------------------------------------------------------------------------

def save_identity_fn(name: str, img1, img2, img3, img4, img5):
    """
    Collect uploaded image paths, call save_identity(), return status string.
    Gradio passes None for slots the user left empty.
    """
    if not name or not name.strip():
        return "Error: Identity name cannot be empty."

    name = name.strip()
    paths = [p for p in [img1, img2, img3, img4, img5] if p is not None]

    if not paths:
        return "Error: Please upload at least one face image."

    try:
        save_identity(name, paths, save_dir=EMBEDDINGS_DIR)
        return f"Identity saved: {name} ({len(paths)} image{'s' if len(paths) > 1 else ''} aggregated)"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tab 2 — Generate
# ---------------------------------------------------------------------------

def refresh_identities():
    names = list_identities(save_dir=EMBEDDINGS_DIR)
    if not names:
        return gr.update(choices=[], value=None)
    return gr.update(choices=sorted(names), value=sorted(names)[0])


def generate_fn(
    identity_name: str,
    prompt: str,
    negative_prompt: str,
    style_name: str,
    num_steps: int,
    guidance_scale: float,
    ip_adapter_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    if not identity_name:
        raise gr.Error("No identity selected. Create one in the 'Create Identity' tab first.")

    if not prompt:
        prompt = "a person"

    # Apply style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    # Load identity
    try:
        record = load_identity_record(identity_name, save_dir=EMBEDDINGS_DIR)
    except FileNotFoundError:
        raise gr.Error(f"Identity '{identity_name}' not found. Please save it first.")

    master_embedding = np.array(record["embedding"], dtype=np.float32)
    kps_source_path = record["source_images"][0]

    # Extract keypoints from first source image
    try:
        face_kps, height, width = extract_kps(kps_source_path)
    except ValueError as e:
        raise gr.Error(str(e))

    # Run pipeline
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=master_embedding,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=ip_adapter_scale,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    ).images[0]

    return image


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

css = ".gradio-container { max-width: 1100px !important; margin: auto; }"

with gr.Blocks(css=css, title="InstantID — Multi-Image Identity") as demo:

    gr.Markdown(
        """
        # InstantID — Multi-Image Identity
        Build a robust identity from multiple photos, then generate images in any style.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Create Identity ──────────────────────────────────────────
        with gr.Tab("Create Identity"):
            gr.Markdown(
                "Upload **1–5 photos** of the same person. "
                "Embeddings are weighted by detection confidence and averaged."
            )

            with gr.Row():
                img_inputs = [
                    gr.Image(
                        label=f"Photo {i + 1}" + (" (required)" if i == 0 else " (optional)"),
                        type="filepath",
                        height=220,
                    )
                    for i in range(5)
                ]

            with gr.Row():
                identity_name_input = gr.Textbox(
                    label="Identity Name",
                    placeholder='e.g. "john" or "alice"',
                    scale=3,
                )
                save_btn = gr.Button("Save Identity", variant="primary", scale=1)

            save_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Status will appear here after saving...",
            )

            save_btn.click(
                fn=save_identity_fn,
                inputs=[identity_name_input] + img_inputs,
                outputs=save_status,
            )

        # ── Tab 2: Generate ─────────────────────────────────────────────────
        with gr.Tab("Generate"):
            with gr.Row():

                # Left column — controls
                with gr.Column(scale=1):
                    with gr.Row():
                        initial_names = sorted(list_identities(save_dir=EMBEDDINGS_DIR))
                        identity_dropdown = gr.Dropdown(
                            label="Select Identity",
                            choices=initial_names,
                            value=initial_names[0] if initial_names else None,
                            scale=3,
                        )
                        refresh_btn = gr.Button("Refresh", scale=1)

                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="a person in a cinematic scene",
                        value="",
                    )
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        value=(
                            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, "
                            "(frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, "
                            "blurry, deformed cat, photo, monochrome, pet collar, gun, weapon"
                        ),
                    )
                    style_input = gr.Dropdown(
                        label="Style Template",
                        choices=STYLE_NAMES,
                        value=DEFAULT_STYLE,
                    )

                    num_steps_slider = gr.Slider(
                        label="Inference Steps",
                        minimum=20, maximum=50, step=1, value=30,
                    )
                    guidance_slider = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0, maximum=12.0, step=0.5, value=5.0,
                    )
                    ip_scale_slider = gr.Slider(
                        label="IP Adapter Scale (identity strength)",
                        minimum=0.1, maximum=1.0, step=0.05, value=0.8,
                    )

                    generate_btn = gr.Button("Generate", variant="primary")

                # Right column — output
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Generated Image", height=512)

            refresh_btn.click(
                fn=refresh_identities,
                outputs=identity_dropdown,
            )

            generate_btn.click(
                fn=generate_fn,
                inputs=[
                    identity_dropdown,
                    prompt_input,
                    negative_prompt_input,
                    style_input,
                    num_steps_slider,
                    guidance_slider,
                    ip_scale_slider,
                ],
                outputs=output_image,
            )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
