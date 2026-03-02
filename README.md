<<<<<<< HEAD

# IPC-Dehaze

AI-powered image dehazing and license-plate enhancement for clearer, actionable visuals from hazy scenes.

## 1. Overview / Introduction

IPC-Dehaze is a deep-learning pipeline that removes haze from images and optionally detects and enhances vehicle license plates. It is designed for scenarios where haze, fog, or atmospheric distortion hides important visual details, such as traffic enforcement, parking monitoring, or forensic analysis. The project provides both a web interface and command-line workflows to make dehazing accessible to non-technical users while still offering full control to advanced users.

The problem it solves is visibility loss in real-world imagery: haze reduces contrast and obscures fine details. IPC-Dehaze addresses this by using a transformer-based dehazing model to reconstruct a clearer image, then running a multi-method plate detector to highlight and refine license plate regions. In simple terms, it “cleans the air” in a photo and then focuses on the plate to make it easier to read.

Assumption: Because you did not provide a custom description, this overview is derived from the codebase and repository structure.

## 2. Features

- **AI dehazing pipeline**: Restores visibility using a transformer + VQGAN model to improve contrast and detail.
- **Automatic license plate detection**: Combines color, edge, OpenCV, and optional YOLO methods for robust detection.
- **Plate enhancement and OCR**: Enhances detected plates and optionally runs OCR for text extraction and validation.
- **Web application**: Flask UI for quick uploads and automatic processing without scripting.
- **Two-phase workflow**: Supports separate dehazing and plate detection for flexible usage.
- **Batch processing**: Handles folders of images in one run for efficiency.

## 3. How It Works (High-Level Workflow)

1. The user provides an image (or a folder of images).
2. The system resizes and pads the image for model compatibility.
3. The dehazing model generates a clearer image.
4. If enabled, the system detects license plates in the dehazed output.
5. Detected plate regions are enhanced; OCR is attempted if available.
6. The system saves a dehazed image and a final enhanced image to the output folder.

## 4. Technical Deep Dive (The "How")

### 4.1. System Architecture

IPC-Dehaze is a pipeline-based application with both CLI and web interfaces.

- **Frontend**: Flask template served from `templates/` for uploads and results.
- **Backend**: Flask app in `app.py` orchestrates uploads and invokes the inference pipeline.
- **Inference Engine**: `inference.py` runs the dehazing model and optional plate detection.
- **Phase 2 Detector**: `detect_plates.py` runs plate detection on already dehazed images.
- **Model Core**: `basicsr/` contains model architecture and utilities.

Communication is local and file-based: the web app saves uploads to disk, calls `inference.py`, and serves generated files from `web_results/`.

### 4.2. Model Architecture (If applicable)

Assumption: Model names are inferred from the codebase because exact names were not provided.

- **Dehazing model**: `DehazeTokenNet` with a VQGAN backbone and a critic network (`Critic`).
- **Why this architecture**: Transformer-style token refinement paired with a VQGAN encoder/decoder is effective for preserving structure while removing haze, and the critic helps refine quality iteratively.
- **Key components**:
  - **VQGAN encoder** for multiscale feature extraction.
  - **Transformer token decoder** for iterative refinement.
  - **Critic network** for quality guidance.
- **Input/Output shapes**:
  - Input: variable-size RGB images, padded to multiples of 32.
  - Output: same spatial dimensions as input (after optional resizing strategy).

Optional detection model:

- **YOLO**: `yolov8n.pt` is used when `ultralytics` is installed and the model file is present.

### 4.3. Working Mechanism / Core Logic

The core logic is a two-stage pipeline:

1. **Dehazing**: Encode image into tokens, iteratively decode and refine, then reconstruct.
2. **Detection + Enhancement**: Find plate regions, enhance them, and optionally OCR the text.

Pseudo-flow:

```
load image
resize/pad -> model input
dehazed = dehaze_model(image)
if enhance_plates:
    plates = detect_plates(dehazed)
    for plate in plates:
        enhance plate region
        ocr text (optional)
save outputs
```

### 4.4. Data Flow

- **Input**: Image files (`.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`).
- **Processing**:
  - Read and normalize image.
  - Resize based on `--max_size`, `--resize_to`, or forced dimensions.
  - Pad to model window size.
  - Dehaze via `DehazeTokenNet` and VQGAN decoder.
  - Optional plate detection and enhancement.
  - Optional OCR with format validation (Indian plate format: XXNNXXNNNN).
- **Output**:
  - `*_dehazed.png` (dehazed-only output).
  - `*.png` (final enhanced output).
  - Optional plate crops (`*_plate_1.png`, etc.) when using `detect_plates.py`.

## 5. Technology Stack

- **Languages**: Python (assumption based on codebase).
- **Frameworks**: Flask.
- **ML / CV Libraries**: PyTorch, OpenCV, BasicSR.
- **Optional ML Tools**: Ultralytics YOLO, EasyOCR.
- **Utilities**: NumPy, tqdm.

Assumption: This list is inferred from the repository imports and requirements files.

## 6. Installation and Setup

1. **Prerequisites**:
   - Python 3.8+ (assumption).
   - GPU with CUDA for faster inference (optional).
2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd IPC-Dehaze
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Add pretrained models**:
   - `pretrained_models/predictor.pth`
   - `pretrained_models/critic.pth`

Assumption: Environment setup uses standard Python venv on Windows; adjust as needed for your platform.

## 7. Usage

### Web app

```bash
python app.py
```

Open `http://localhost:5000` and upload images.

### CLI (full pipeline)

```bash
python inference.py -i examples -o results --enhance_plates
```

### CLI (dehaze only)

```bash
python inference.py -i examples -o results --dehaze_only
```

### Two-phase usage

```bash
python inference.py -i examples -o results --dehaze_only
python detect_plates.py -i results -o results_user
```

## 8. Results and Performance (Optional but recommended)

Assumption: No benchmark metrics were provided. Suggested placeholders:

- **Model Quality**: Not measured (PSNR/SSIM not reported).
- **Inference Speed**: Not measured (depends on GPU/CPU).

[Image: Example dehazed output with plate enhancement]

## 9. Future Work / Roadmap

- Add quantitative benchmarks (PSNR/SSIM) and runtime profiling.
- Train or integrate a dedicated license plate YOLO model for higher recall.
- Add a batch-results viewer and CSV export of OCR outputs.

## 10. Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request with a clear description.
