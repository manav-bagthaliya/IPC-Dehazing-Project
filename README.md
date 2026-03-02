# IPC-Dehaze: AI-Powered Image Dehazing & License Plate Enhancement

**Transform hazy images into clear, actionable visuals with advanced deep learning**

![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Model Details](#model-details)
- [License Plate Detection](#license-plate-detection)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🌟 Overview

**IPC-Dehaze** (Image Processing and Computer Vision - Dehazing) is a state-of-the-art deep learning system designed to restore visibility in hazy, foggy, or atmospheric-distorted images. The system combines **transformer-based dehazing** with **multi-method license plate detection** to deliver clear, readable results for:

- 🚓 **Traffic enforcement** and monitoring
- 🅿️ **Parking management** systems
- 🔍 **Forensic analysis** and investigation
- 🌫️ **Environmental monitoring** in poor visibility conditions
- 📸 **Surveillance systems** enhancement

### The Problem

Haze, fog, and atmospheric particles scatter light, reducing image contrast and obscuring critical details like license plates, signs, and objects. This makes visual identification nearly impossible in traffic monitoring, security, and forensic applications.

### The Solution

IPC-Dehaze uses **DehazeTokenNet** with VQGAN architecture to reconstruct clear images from hazy inputs, followed by intelligent license plate detection using multiple computer vision techniques. The system "cleans the air" in your photos and focuses on extracting actionable information.

---

## ✨ Key Features

### 🎯 Core Capabilities

- ** AI-Powered Dehazing**: Transformer + VQGAN architecture for superior haze removal
- ** Multi-Method Plate Detection**: Combines color-based, edge detection, OpenCV, and YOLO methods
- ** Smart Plate Enhancement**: Automatic detection and enhancement of license plate regions
- ** OCR Integration**: Text extraction with Indian license plate format validation
- ** Web Interface**: User-friendly Flask-based UI for non-technical users
- ** CLI Support**: Full command-line control for batch processing and automation
- ** Two-Phase Workflow**: Flexible separation of dehazing and plate detection
- ** Batch Processing**: Process entire folders of images efficiently

### 🎨 Advanced Features

- **Multiple plate types support**: Yellow (commercial), White (private), Blue (government)
- **Iterative refinement**: Configurable dehazing iterations for quality control
- **Flexible resizing**: Multiple strategies for image preprocessing
- **GPU acceleration**: CUDA support for faster processing
- **Session management**: Track multiple processing jobs via web interface
- **Comprehensive logging**: Detailed processing information and error handling

---

## 🔄 How It Works

### High-Level Workflow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input     │───▶│   Resize &   │───▶│    Dehaze       │───▶│  Plate Detect   │
│   Image     │    │     Pad      │    │  (DehazeNet)    │    │  & Enhance      │
└─────────────┘    └──────────────┘    └─────────────────┘    └─────────────────┘
                                                                         │
                   ┌─────────────────────────────────────────────────────┘
                   ▼
         ┌──────────────────┐    ┌──────────────┐    ┌─────────────────┐
         │  Enhanced Image  │◀───│  OCR Text    │◀───│  Plate Crops    │
         │    + Metadata    │    │  Validation  │    │   (Optional)    │
         └──────────────────┘    └──────────────┘    └─────────────────┘
```

### Detailed Pipeline

1. **Input Processing**
   - Load image(s) from file or upload
   - Validate format and dimensions
   - Apply resizing strategy based on parameters

2. **Preprocessing**
   - Normalize pixel values
   - Pad to model-compatible dimensions (multiples of 32/64)
   - Convert to tensor format

3. **Dehazing** _(Phase 1)_
   - Encode image using VQGAN encoder
   - Process through transformer token predictor
   - Iterative refinement with critic network
   - Decode to clear image

4. **License Plate Detection** _(Phase 2 - Optional)_
   - **Color Detection**: HSV-based detection for yellow/white/blue plates
   - **Edge Detection**: Canny + contour analysis
   - **OpenCV Methods**: Cascade classifiers and morphological operations
   - **YOLO Detection**: Deep learning-based detection (if available)
   - Filter and merge overlapping detections

5. **Enhancement & OCR**
   - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Sharpen plate regions
   - Extract text using EasyOCR
   - Validate against Indian plate format (XX NN XX NNNN)

6. **Output Generation**
   - Save dehazed image (`*_dehazed.png`)
   - Save final enhanced image (`*.png`)
   - Save plate crops with metadata (`*_plate_N.png`)
   - Generate processing reports

---

## 🏗️ Architecture

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         User Interface                        │
│  ┌─────────────────┐              ┌────────────────────────┐ │
│  │  Web Interface  │              │   CLI Interface        │ │
│  │   (Flask App)   │              │  (argparse + scripts)  │ │
│  └────────┬────────┘              └───────────┬────────────┘ │
└───────────┼──────────────────────────────────┼───────────────┘
            │                                   │
            └───────────┬───────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                    Processing Engine                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │               inference.py (Main Pipeline)              │ │
│  │  • Image loading & preprocessing                        │ │
│  │  • Model inference & dehazing                           │ │
│  │  • Integrated plate detection                           │ │
│  └──────────────────────┬──────────────────────────────────┘ │
│                         │                                     │
│  ┌─────────────────────┴──────────────────────────────────┐ │
│  │          detect_plates.py (Phase 2 Detector)           │ │
│  │  • Advanced plate detection on dehazed images          │ │
│  │  • Multiple detection methods                           │ │
│  │  • OCR and validation                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      Model Components                         │
│  ┌───────────────┐  ┌────────────┐  ┌────────────────────┐  │
│  │ DehazeTokenNet│  │   Critic   │  │ Detection Models   │  │
│  │  (Predictor)  │  │  Network   │  │ (YOLO + OpenCV)    │  │
│  └───────┬───────┘  └─────┬──────┘  └──────────┬─────────┘  │
│          └────────────────┴────────────────────┘             │
│                            │                                  │
│                    ┌───────▼────────┐                        │
│                    │  VQGAN Encoder │                        │
│                    │    & Decoder   │                        │
│                    └────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### Model Architecture: DehazeTokenNet

```
Input Image
      │
      ▼
┌─────────────────┐
│ VQGAN Encoder   │──▶ Multi-scale features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Token Encoding │──▶ Quantized tokens (H/64 × W/64 × 256)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Predictor (Transformer)          │
│  • 4× RSTB blocks (Swin Transformer)│
│  • Self-attention layers            │
│  • Codebook prediction              │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Critic Network │──▶ Quality assessment & refinement
│  • 2× RSTB blocks│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VQGAN Decoder   │──▶ Reconstructed clear image
└─────────────────┘
```

**Key Components:**

- **VQGAN Encoder/Decoder**: Vector Quantized GAN for high-quality image reconstruction
- **Predictor**: Swin Transformer-based token predictor with 4 RSTB blocks
- **Critic**: Quality assessment network with 2 RSTB blocks
- **Codebook**: 1024 embeddings of 256 dimensions each
- **Feature Resolution**: 64× downsampling for efficient processing

---

## 💻 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended for large images
- **Storage**: ~5GB for models and dependencies

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd IPC-Dehaze
   ```

2. **Create Virtual Environment**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pretrained Models**

   Place the following model files in `pretrained_models/`:
   - `predictor.pth` - Main dehazing model
   - `critic.pth` - Quality refinement model

   ```bash
   # Create directory if it doesn't exist
   mkdir -p pretrained_models

   # Download models (use provided download script)
   python scripts/download_pretrained_models.py
   ```

5. **Verify Installation**
   ```bash
   python setup.py
   ```
   This will check Python version, dependencies, CUDA availability, and create necessary directories.

### Optional Components

**For YOLO-based detection:**

```bash
pip install ultralytics
```

**For OCR functionality:**

```bash
pip install easyocr
```

---

## 🚀 Usage

### Web Interface (Recommended for Beginners)

1. **Start the Flask Server**

   ```bash
   python app.py
   ```

2. **Open Browser**
   Navigate to `http://localhost:5000`

3. **Upload & Process**
   - Upload up to 10 images (max 100MB total)
   - Adjust processing parameters:
     - **Iterations**: 1-20 (default: 8)
     - **Max Size**: Maximum dimension (default: 1500)
     - **Enhance Plates**: Enable/disable plate detection
   - Click "Process" and download results

**Web Interface Features:**

- Real-time progress tracking
- Image preview before/after
- Batch download of results
- Session-based processing
- Automatic plate detection

### Command-Line Interface (CLI)

#### Basic Dehazing

```bash
python inference.py -i examples -o results
```

#### Dehazing + License Plate Detection

```bash
python inference.py -i examples -o results --enhance_plates
```

#### Dehaze Only (No Plate Detection)

```bash
python inference.py -i examples -o results --dehaze_only
```

#### Advanced Options

```bash
python inference.py \
  -i input_folder \
  -o output_folder \
  -n 10 \
  --max_size 2000 \
  --resize_to 1024 \
  --enhance_plates \
  --save_plate_crops
```

**CLI Parameters:**

- `-i, --input`: Input image or folder path
- `-o, --output`: Output folder path
- `-n, --num_iterations`: Number of dehazing iterations (default: 8)
- `--max_size`: Maximum dimension for resizing (default: 1500)
- `--resize_to`: Force resize to specific dimension
- `--force_width`, `--force_height`: Force specific dimensions
- `--enhance_plates`: Enable automatic license plate detection
- `--dehaze_only`: Skip plate detection
- `--save_plate_crops`: Save individual plate crops

### Two-Phase Workflow

For better control, separate dehazing and plate detection:

**Phase 1: Dehaze Images**

```bash
python inference.py -i raw_images -o dehazed_output --dehaze_only
```

**Phase 2: Detect Plates**

```bash
python detect_plates.py -i dehazed_output -o final_results
```

This approach is useful for:

- Processing large batches
- Fine-tuning plate detection separately
- Reprocessing already dehazed images
- Debugging and testing

### Example Workflows

**Scenario 1: Quick Processing**

```bash
# Process single image with defaults
python inference.py -i hazy_image.jpg -o results
```

**Scenario 2: High-Quality Processing**

```bash
# Maximum quality with more iterations
python inference.py -i input/ -o output/ -n 15 --enhance_plates
```

**Scenario 3: Large Image Batch**

```bash
# Process efficiently with size limits
python inference.py -i large_batch/ -o results/ --max_size 1200 --dehaze_only
python detect_plates.py -i results/ -o final/
```

## 🛠️ Technology Stack

### Core Technologies

| Category             | Technologies                           |
| -------------------- | -------------------------------------- |
| **Language**         | Python 3.8+                            |
| **Web Framework**    | Flask 2.3+                             |
| **Deep Learning**    | PyTorch 1.12+, torchvision, torchaudio |
| **Computer Vision**  | OpenCV 4.8+, BasicSR                   |
| **Model Components** | timm, einops                           |
| **Object Detection** | Ultralytics YOLO 8.0+ (optional)       |
| **OCR**              | EasyOCR 1.7+ (optional)                |
| **Utilities**        | NumPy, tqdm, Werkzeug                  |

### Development Tools

- **Version Control**: Git
- **Dependency Management**: pip, virtualenv
- **Training**: PyTorch distributed, CUDA
- **Monitoring**: tqdm for progress bars

---

## 🧠 Model Details

### DehazeTokenNet Architecture

**Paper/Method**: Based on FeMaSR (Feature Matching Super-Resolution) adapted for dehazing

**Model Characteristics:**

- **Type**: Transformer-based generative model with VQGAN
- **Input**: RGB images (any size, automatically padded)
- **Output**: Clear dehazed RGB images
- **Codebook**: 1024 tokens, 256-dimensional embeddings
- **Downsampling**: 64× (from 256×256 to 4×4 tokens)
- **Parameters**: ~50M (estimated)

**Training Configuration:**

- **Stage 1**: VQGAN pretraining on high-quality images
- **Stage 2**: Predictor training with synthetic haze
- **Stage 3**: Critic training for quality refinement
- **Dataset**: Custom haze synthesis from RGB + depth images
- **Batch Size**: 4 per GPU
- **Learning Rate**: 1e-4 (predictor), 4e-4 (discriminator)
- **Optimizer**: Adam (β₁=0.9, β₂=0.99)

**Haze Synthesis Parameters:**

- Beta range: [0.3, 1.5] (scattering coefficient)
- Atmospheric light (A): [0.25, 1.0]
- Color variations: [-0.025, 0.025]

---

## 🚗 License Plate Detection

### Detection Methods

The system employs **5 complementary detection methods** for maximum recall:

#### 1. **Yellow Plate Detection** (Commercial Vehicles)

- HSV color space filtering
- Range: H: [15-35], S: [80-255], V: [80-255]
- Morphological operations for noise removal
- Aspect ratio filtering: 2:1 to 5:1

#### 2. **White Plate Detection** (Private Vehicles)

- Adaptive HSV thresholding
- Enhanced sensitivity: H: [0-180], S: [0-50], V: [180-255]
- Minimum area: 1000 pixels
- Convex hull analysis

#### 3. **Enhanced White Detection**

- Secondary pass with relaxed parameters
- Catches plates missed by strict filtering
- Gradient-based enhancement

#### 4. **Blue Plate Detection** (Government Vehicles)

- Blue color space targeting
- Range: H: [90-130], S: [80-255], V: [80-255]
- Special handling for government vehicles

#### 5. **OpenCV Traditional Detection**

- Edge detection (Canny)
- Contour analysis
- Morphological transformations
- Aspect ratio and area validation

#### 6. **YOLO Detection** (Optional)

- YOLOv8 nano model
- Real-time object detection
- High accuracy for visible plates
- Requires: `pip install ultralytics`

### OCR & Validation

**Indian License Plate Format**: `XX NN XX NNNN`

- State Code: 2 letters (e.g., MH, DL, KA)
- District Code: 2 digits (e.g., 01, 02)
- Series: 2 letters (e.g., AB, CD)
- Unique Number: 4 digits (e.g., 1234)

**Validation Features:**

- Format checking with detailed error reporting
- Confidence scoring based on valid segments
- Character correction for common OCR errors
- Formatted output: "MH 01 AB 1234"

### Enhancement Pipeline

For each detected plate:

1. **Contrast Enhancement**: CLAHE with clip limit 2.0
2. **Sharpening**: Unsharp mask for edge clarity
3. **Noise Reduction**: Bilateral filtering
4. **Binarization**: Adaptive thresholding for OCR
5. **Text Extraction**: EasyOCR with English model
6. **Validation**: Format checking and confidence scoring

---

## 🤝 Contributing

We welcome contributions to improve IPC-Dehaze! Here's how you can help:

### Ways to Contribute

1. **Bug Reports**: Open an issue with detailed reproduction steps
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests for bug fixes or features
4. **Documentation**: Improve README, add tutorials, or create examples
5. **Testing**: Test on different platforms and report results

### Development Workflow

1. **Fork the Repository**

   ```bash
   git fork <repository-url>
   ```

2. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests if applicable

4. **Test Your Changes**

   ```bash
   python setup.py  # Verify installation
   python inference.py -i examples -o test_output  # Test functionality
   ```

5. **Commit Changes**

   ```bash
   git add .
   git commit -m "Add: detailed description of changes"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request with:
   - Clear description of changes
   - Screenshots/examples if applicable
   - Related issue numbers

### Code Standards

- **Python**: PEP 8 compliant
- **Documentation**: Docstrings for all functions and classes
- **Comments**: Explain complex logic
- **Testing**: Test new features before submitting

---

## 📄 License

This project is licensed under the **Pi-Lab License 1.0** - see the [LICENSE.txt](LICENSE.txt) file for details.

**Key Points:**

- ✅ Free for non-commercial research and educational use
- ✅ Open source with attribution required
- ❌ Commercial use requires written permission
- ❌ No warranty or liability

Copyright © 2024 Pi-Lab. All rights reserved.

---

## 🙏 Acknowledgments

### Research Foundations

This project builds upon several key research works:

- **FeMaSR**: Feature Matching Super-Resolution methodology
- **VQGAN**: Vector Quantized Generative Adversarial Networks
- **Swin Transformer**: Hierarchical Vision Transformer
- **BasicSR**: Basic Super-Resolution framework

### Libraries & Tools

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Ultralytics**: YOLOv8 implementation
- **EasyOCR**: Optical character recognition
- **Flask**: Web application framework

### Community

Thanks to all contributors, testers, and users who help improve this project!

---

## 📞 Support & Contact

- **Email**: [22it005@charusat.edu.in]

---

### Version History

- **v1.0.0** (Current): Initial release with dehazing and plate detection
- **v0.9.0**: Beta testing phase
- **v0.5.0**: Core model implementation

_"Clearing the haze, one image at a time"_
