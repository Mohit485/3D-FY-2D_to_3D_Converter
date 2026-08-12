import os
import cv2
import torch
import tempfile
import subprocess
import numpy as np
import gradio as gr
import spaces

import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_INPUT_SIZE = (640, 480)
TEMPORAL_SMOOTHING = 0.75

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD MiDaS
# ============================================================

print("Loading MiDaS DPT-Hybrid...")

model = torch.hub.load(
    "intel-isl/MiDaS",
    "DPT_Hybrid",
    trust_repo=True
)

model.to(DEVICE)
model.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms",
    trust_repo=True
)

transform = midas_transforms.dpt_transform

print("MiDaS loaded successfully.")


# ============================================================
# DEPTH ESTIMATION
# ============================================================

def get_depth(frame, previous_depth=None):
    """
    Estimate a normalized depth map for one video frame.

    Temporal smoothing is applied using the previous frame's
    depth map to reduce frame-to-frame flickering.
    """

    small = cv2.resize(
        frame,
        MODEL_INPUT_SIZE
    )

    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = transform(rgb).to(DEVICE)

    with torch.no_grad():

        depth = model(input_tensor)

        depth = F.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth = depth.cpu().numpy()

    # Normalize depth to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()

    depth = (
        depth - depth_min
    ) / (
        depth_max - depth_min + 1e-8
    )

    # Temporal smoothing
    if previous_depth is not None:

        depth = (
            TEMPORAL_SMOOTHING * depth
            +
            (1 - TEMPORAL_SMOOTHING) * previous_depth
        )

    return depth


# ============================================================
# PIXEL SHIFTING
# ============================================================

def shift_pixels(frame, depth, direction, strength):

    h, w = frame.shape[:2]

    shift = (
        depth * strength * direction
    ).astype(np.int32)

    ys, xs = np.mgrid[0:h, 0:w]

    new_xs = np.clip(
        xs + shift,
        0,
        w - 1
    )

    output = np.zeros_like(frame)

    output[
        ys,
        new_xs
    ] = frame[
        ys,
        xs
    ]

    return output


# ============================================================
# ANAGLYPH GENERATION
# ============================================================

def make_anaglyph(frame, depth, strength):

    left = shift_pixels(
        frame,
        depth,
        +1,
        strength
    )

    right = shift_pixels(
        frame,
        depth,
        -1,
        strength
    )

    anaglyph = np.zeros_like(frame)

    # OpenCV uses BGR
    anaglyph[:, :, 2] = left[:, :, 2]   # Red
    anaglyph[:, :, 1] = right[:, :, 1]  # Green
    anaglyph[:, :, 0] = right[:, :, 0]  # Blue

    return anaglyph


# ============================================================
# SIDE-BY-SIDE GENERATION
# ============================================================

def make_sbs(frame, depth, strength):

    left = shift_pixels(
        frame,
        depth,
        +1,
        strength
    )

    right = shift_pixels(
        frame,
        depth,
        -1,
        strength
    )

    sbs = np.concatenate(
        [left, right],
        axis=1
    )

    return sbs


# ============================================================
# AUDIO REMUXING
# ============================================================

def add_audio(
    silent_video,
    original_video,
    output_video
):
    """
    Add the original video's audio track to the
    generated silent video using FFmpeg.
    """

    command = [
        "ffmpeg",
        "-y",

        "-i",
        silent_video,

        "-i",
        original_video,

        "-map",
        "0:v:0",

        "-map",
        "1:a:0?",

        "-c:v",
        "libx264",

        "-preset",
        "fast",

        "-crf",
        "23",

        "-c:a",
        "aac",

        "-shortest",

        output_video
    ]

    try:

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if os.path.exists(output_video):
            return output_video

    except subprocess.CalledProcessError as e:

        print(
            "FFmpeg error:",
            e.stderr.decode(
                errors="ignore"
            )
        )

    return None


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

@spaces.GPU
def process(
    input_video,
    output_choice,
    strength,
    max_seconds
):
    """
    Convert a normal 2D video into depth-based 3D outputs.

    Parameters
    ----------
    input_video:
        Uploaded video file.

    output_choice:
        Anaglyph, Side-by-Side, Depth Map, or All.

    strength:
        Controls the stereo pixel displacement.

    max_seconds:
        Maximum amount of video to process.

    Returns
    -------
    tuple
        Generated Anaglyph, SBS and Depth Map video paths.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if input_video is None:
        return None, None, None

    strength = int(strength)
    max_seconds = int(max_seconds)

    print(
        f"Starting processing: "
        f"{output_choice}, "
        f"strength={strength}, "
        f"max_seconds={max_seconds}"
    )

    # --------------------------------------------------------
    # Temporary output directory
    # --------------------------------------------------------

    output_dir = tempfile.mkdtemp(
        prefix="3dfy_"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        input_video
    )

    if not cap.isOpened():
        print("Could not open input video.")
        return None, None, None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 24.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if width <= 0 or height <= 0:

        cap.release()

        print("Invalid video dimensions.")

        return None, None, None

    max_frames = int(
        fps * max_seconds
    )

    print(
        f"Video: {width}x{height} @ {fps:.2f} FPS"
    )

    print(
        f"Processing up to {max_frames} frames"
    )

    # --------------------------------------------------------
    # Determine outputs
    # --------------------------------------------------------

    generate_anaglyph = (
        output_choice
        in ["Anaglyph", "All"]
    )

    generate_sbs = (
        output_choice
        in ["Side-by-Side", "All"]
    )

    generate_depth = (
        output_choice
        in ["Depth Map", "All"]
    )

    # --------------------------------------------------------
    # Create output paths
    # --------------------------------------------------------

    ana_path = os.path.join(
        output_dir,
        "anaglyph.mp4"
    )

    sbs_path = os.path.join(
        output_dir,
        "sidebyside.mp4"
    )

    depth_path = os.path.join(
        output_dir,
        "depth.mp4"
    )

    # --------------------------------------------------------
    # Video codec
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    # --------------------------------------------------------
    # Create writers only when needed
    # --------------------------------------------------------

    out_ana = None
    out_sbs = None
    out_depth = None

    if generate_anaglyph:

        out_ana = cv2.VideoWriter(
            ana_path,
            fourcc,
            fps,
            (width, height)
        )

    if generate_sbs:

        out_sbs = cv2.VideoWriter(
            sbs_path,
            fourcc,
            fps,
            (width * 2, height)
        )

    if generate_depth:

        out_depth = cv2.VideoWriter(
            depth_path,
            fourcc,
            fps,
            (width, height)
        )

    # --------------------------------------------------------
    # Temporal depth state
    # --------------------------------------------------------

    previous_depth = None

    frame_number = 0

    # --------------------------------------------------------
    # Frame processing loop
    # --------------------------------------------------------

    try:

        while frame_number < max_frames:

            ret, frame = cap.read()

            if not ret:
                break

            # --------------------------------------------
            # MiDaS depth estimation
            # --------------------------------------------

            depth = get_depth(
                frame,
                previous_depth
            )

            previous_depth = depth

            # --------------------------------------------
            # Anaglyph
            # --------------------------------------------

            if generate_anaglyph:

                anaglyph = make_anaglyph(
                    frame,
                    depth,
                    strength
                )

                out_ana.write(
                    anaglyph
                )

            # --------------------------------------------
            # Side-by-Side
            # --------------------------------------------

            if generate_sbs:

                sbs = make_sbs(
                    frame,
                    depth,
                    strength
                )

                out_sbs.write(
                    sbs
                )

            # --------------------------------------------
            # Depth Map
            # --------------------------------------------

            if generate_depth:

                depth_uint8 = (
                    depth * 255
                ).astype(
                    np.uint8
                )

                depth_colored = cv2.applyColorMap(
                    depth_uint8,
                    cv2.COLORMAP_MAGMA
                )

                out_depth.write(
                    depth_colored
                )

            # --------------------------------------------
            # Progress
            # --------------------------------------------

            frame_number += 1

            if frame_number % 20 == 0:

                progress = (
                    frame_number /
                    max_frames
                ) * 100

                print(
                    f"Processed "
                    f"{frame_number}/{max_frames} "
                    f"({progress:.1f}%)"
                )

    finally:

        cap.release()

        if out_ana is not None:
            out_ana.release()

        if out_sbs is not None:
            out_sbs.release()

        if out_depth is not None:
            out_depth.release()

    print("Video processing completed.")

    # --------------------------------------------------------
    # Add original audio
    # --------------------------------------------------------

    final_ana = None
    final_sbs = None
    final_depth = None

    if generate_anaglyph:

        final_ana = os.path.join(
            output_dir,
            "anaglyph_final.mp4"
        )

        final_ana = add_audio(
            ana_path,
            input_video,
            final_ana
        )

    if generate_sbs:

        final_sbs = os.path.join(
            output_dir,
            "sbs_final.mp4"
        )

        final_sbs = add_audio(
            sbs_path,
            input_video,
            final_sbs
        )

    if generate_depth:

        final_depth = os.path.join(
            output_dir,
            "depth_final.mp4"
        )

        final_depth = add_audio(
            depth_path,
            input_video,
            final_depth
        )

    print("All requested outputs generated.")

    return (
        final_ana,
        final_sbs,
        final_depth
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="3D-FY",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🎬 3D-FY — AI 2D to 3D Video Converter

        Convert normal 2D videos into immersive 3D formats
        using **MiDaS DPT-Hybrid depth estimation**.

        Designed for **VR headsets** and **red-cyan 3D glasses**.
        """
    )

    with gr.Row():

        # ====================================================
        # INPUT / SETTINGS
        # ====================================================

        with gr.Column():

            gr.Markdown(
                "### ⚙️ Settings"
            )

            input_video = gr.Video(
                label="Upload 2D Video",
                sources=["upload"],
                type="filepath"
            )

            output_choice = gr.Dropdown(
                choices=[
                    "Anaglyph",
                    "Side-by-Side",
                    "Depth Map",
                    "All"
                ],
                value="All",
                label="Output Format"
            )

            strength = gr.Slider(
                minimum=5,
                maximum=50,
                value=25,
                step=5,
                label="3D Effect Strength"
            )

            max_seconds = gr.Slider(
                minimum=5,
                maximum=60,
                value=15,
                step=5,
                label="Maximum Processing Time (seconds)"
            )

            convert_button = gr.Button(
                "🚀 Convert to 3D",
                variant="primary"
            )

        # ====================================================
        # OUTPUTS
        # ====================================================

        with gr.Column():

            gr.Markdown(
                "### 📤 Generated Outputs"
            )

            output_anaglyph = gr.Video(
                label="Anaglyph — Red/Cyan Glasses"
            )

            output_sbs = gr.Video(
                label="Side-by-Side — VR Headset"
            )

            output_depth = gr.Video(
                label="Depth Map Visualization"
            )

    # ========================================================
    # INFORMATION
    # ========================================================

    gr.Markdown(
        """
        ---

        ### How to use

        1. Upload a 2D video.
        2. Select the desired output format.
        3. Adjust the 3D effect strength.
        4. Select the maximum processing duration.
        5. Click **Convert to 3D**.

        ### Output formats

        - **Anaglyph** → Red/Cyan 3D glasses
        - **Side-by-Side** → VR headset
        - **Depth Map** → Visualization of estimated scene depth

        ### Technology

        **MiDaS DPT-Hybrid · PyTorch · OpenCV · Gradio · FFmpeg**
        """
    )

    # ========================================================
    # EVENT
    # ========================================================

    convert_button.click(
        fn=process,
        inputs=[
            input_video,
            output_choice,
            strength,
            max_seconds
        ],
        outputs=[
            output_anaglyph,
            output_sbs,
            output_depth
        ],
        concurrency_limit=1
    )


# ============================================================
# LAUNCH
# ============================================================

demo.queue(
    default_concurrency_limit=1
)

demo.launch()