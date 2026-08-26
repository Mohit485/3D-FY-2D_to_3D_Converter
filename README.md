# 🎬 3D-FY — AI-Powered 2D to 3D Video Converter
Convert any standard 2D video into immersive 3D formats using monocular depth estimation. Outputs are ready for Red-Cyan glasses, VR headsets and 3D visualization — now available as a live web app, with the original free Kaggle GPU version also preserved.

## Try It Live

🔗 **[3D-FY on Hugging Face Spaces](https://huggingface.co/spaces/Madiy/3D-FY)** — no setup, no GPU needed, runs on ZeroGPU

## Kaggle Notebook

View and run the complete project on Kaggle:

🔗[3D-fy-Kaggle-notebook-link](https://www.kaggle.com/code/mohiadiy/3d-fy-ai-driven-2d-to-3d-converter/edit)

-----
# Sample Output
| ![input - anaglyph comparison](assets/Original.gif) |  
| ![input - depth comparison](assets/narcomp.gif) |
-----------------------
  ## How It Works
Input Video → Frame Extraction → MiDaS Depth Estimation → Pixel Shifting → Stereoscopic Synthesis → Output Video

1. Each frame is passed through **MiDaS DPT-Hybrid**, a transformer-based 
   monocular depth estimation model by Intel ISL
2. The depth map is used to horizontally shift pixels — close objects shift 
   more, far objects shift less — simulating the parallax seen by human eyes
3. Left and right eye views are composited into Anaglyph or SBS format
4. Temporal smoothing via exponential moving average reduces depth 
   flickering between frames
5. Original audio is preserved using ffmpeg
-------------------------
## 🖥️ Interface

The project runs as a Gradio web interface, deployed two ways:

- **Hugging Face Spaces (ZeroGPU)** — a persistent, always-available public app
- **Kaggle Notebook** — runs on a free T4 GPU and generates a temporary public link, no local GPU needed

![interface screenshot](assets/interface_ss.png)
--------------------------
## Tech Stack

| Tool | Purpose |
|---|---|
| MiDaS DPT-Hybrid | Monocular depth estimation |
| PyTorch | Model inference |
| OpenCV | Frame read/write, color mapping |
| NumPy | Vectorized pixel shifting |
| Gradio | Interactive web interface |
| ffmpeg | Audio preservation |
| Hugging Face Spaces (ZeroGPU) | Persistent live deployment |
| Kaggle T4 GPU | Free compute (notebook version) |

---------------------------
## Run It Yourself

### Option 1 — Hugging Face Spaces (recommended)
Just open the live link above — no setup required.

### Option 2 — Kaggle Notebook
This project is also designed to run on **Kaggle Notebooks** with a free T4 GPU.

1. Go to [kaggle.com](https://kaggle.com) and create a free account
2. Create a new notebook and enable GPU: 
   `Settings → Accelerator → GPU T4`
3. Upload your video using `Add Data → Upload`
4. Copy each cell from the notebook in order and run them sequentially
5. For manual mode: set `VIDEO_PATH` and run the processing cell
6. Download outputs from `/kaggle/working/output/`
7. For interface mode: run the Gradio cell, open the public link printed in output
---------------------------
## 📤 Output Formats

|    Format    |           View With                  |      Use Case           |
|--------------|--------------------------------------|-------------------------|
|   Anaglyph   |       Red-Cyan 3D glasses            | Cinema, presentations   |
| Side-by-Side | VR headset (Meta Quest, Cardboard)   | Immersive VR viewing    |
|   Depth Map  |          Any player                  | Visualization, debugging|


-----------------------------

## Limitations and Future Scope

- Depth estimation is relative, not metric — transparent surfaces and 
  fast motion create artifacts
- Occlusion hole filling using depth-guided inpainting is under active 
  development 
- Temporal consistency currently uses EMA smoothing — proper video-aware 
  depth models are identified as future work
- Real-time inference requires model optimization (TensorRT, ONNX export)
- On Hugging Face Spaces, processing is capped to fit within ZeroGPU's 
  fixed execution window — longer clips are trimmed to a shorter length

-----------------------------

## 📄 Research Paper

This project is accompanied by a research survey paper covering the history of
depth estimation, stereoscopic synthesis methods, and comparison of existing pipelines.

> Paper title: * 3D-Fy: AI-Based 2D to 3D Video Conversion Using Monocular Depth Estimation*
> Status: Under preparation for IEEE conference submission

---

## Author

**Mohit Aditya**  
B.Tech Artificial Intelligence and Machine Learning  
JB Institute of Technology, Dehradun  
[LinkedIn](https://www.linkedin.com/in/mohit-aditya-55506b255/) 
[GitHub](https://github.com/Mohit485)
[Notion](https://app.notion.com/p/Mohit-Aditya-37ed29052bd9801d80adec7e68a77ef4?source=copy_link)
-----------------------------

## Acknowledgements

- [MiDaS](https://github.com/isl-org/MiDaS) by Intel ISL for the depth 
  estimation model
- [StereoCrafter](https://github.com/TencentARC/StereoCrafter) paper for 
  establishing the problem benchmark
- Kaggle for free GPU compute
- Hugging Face for ZeroGPU Spaces hosting
