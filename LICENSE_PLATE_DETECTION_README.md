# License Plate Detection & Enhancement

This document explains how to use the new license plate detection and enhancement feature in IPC-Dehaze.

## Overview

The license plate detection feature automatically identifies license plates in hazy images and applies special enhancement techniques to improve their readability. This is particularly useful for:

- Traffic monitoring systems
- Law enforcement applications
- Security and surveillance
- Autonomous driving systems
- Parking management

## Features

### 1. **Dual Detection Methods**
- **YOLO-based detection**: Uses pre-trained YOLO models for accurate detection
- **OpenCV-based detection**: Fallback method using traditional computer vision techniques
- **Automatic fallback**: If YOLO fails, automatically switches to OpenCV method

### 2. **Smart Enhancement**
- **Contrast enhancement**: Increases brightness and contrast for better text visibility
- **Sharpening**: Applies edge-preserving sharpening filters
- **Noise reduction**: Uses bilateral filtering to reduce noise while preserving edges
- **OCR preparation**: Optimizes images for text recognition

### 3. **Duplicate Prevention**
- **Overlap detection**: Prevents multiple detections of the same license plate
- **Confidence ranking**: Prioritizes higher-confidence detections
- **Smart merging**: Combines results from multiple detection methods

## Installation

### Required Dependencies

```bash
# Core dependencies (already in requirements.txt)
pip install -r requirements.txt

# Additional dependencies for license plate detection
pip install ultralytics>=8.0.0
pip install easyocr>=1.7.0
```

### Optional Dependencies

```bash
# For better YOLO performance
pip install torch torchvision

# For GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Command Line Interface

```bash
# Basic usage with license plate enhancement
python inference.py -i examples -o results --enhance_plates

# Advanced usage with custom parameters
python inference.py \
    -i examples \
    -o results \
    -n 12 \
    --max_size 2000 \
    --enhance_plates
```

### Web Interface

1. **Upload Images**: Drag and drop your hazy images
2. **Enable License Plate Enhancement**: Check the "Enhance License Plates" checkbox
3. **Configure Parameters**: Adjust other processing options as needed
4. **Process**: Click "Process Images" to start
5. **Download Results**: Get your dehazed images with enhanced license plates

## Configuration Options

### Detection Parameters

```python
# Confidence threshold for YOLO detection
confidence_threshold = 0.5

# Minimum area for OpenCV detection
min_plate_area = 1000

# Aspect ratio range for license plates
min_aspect_ratio = 2.0
max_aspect_ratio = 5.5
```

### Enhancement Parameters

```python
# Contrast enhancement
alpha = 1.8  # Contrast multiplier
beta = 40    # Brightness offset

# Sharpening kernel
sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# Bilateral filter parameters
bilateral_d = 9      # Diameter of pixel neighborhood
bilateral_sigma_color = 75    # Filter sigma in color space
bilateral_sigma_space = 75    # Filter sigma in coordinate space
```

## How It Works

### 1. **Detection Phase**
```
Input Image → YOLO Detection → OpenCV Fallback → Merge Results
```

### 2. **Enhancement Phase**
```
Detected Plate → Contrast Enhancement → Sharpening → Noise Reduction → OCR Prep
```

### 3. **Integration Phase**
```
Enhanced Plate → Replace in Dehazed Image → Final Output
```

## Performance Considerations

### **Speed Optimization**
- **GPU acceleration**: Use CUDA-enabled PyTorch for faster YOLO inference
- **Batch processing**: Process multiple images simultaneously
- **Model caching**: YOLO models are cached after first load

### **Memory Usage**
- **Image resizing**: Large images are automatically resized to prevent memory issues
- **Batch size**: Adjust based on available GPU memory
- **Model selection**: Use smaller YOLO models for faster inference

## Troubleshooting

### Common Issues

1. **No plates detected**
   - Check image quality and lighting
   - Adjust confidence thresholds
   - Verify license plate visibility

2. **Multiple false positives**
   - Increase confidence threshold
   - Adjust aspect ratio constraints
   - Use more specific training data

3. **Poor enhancement quality**
   - Check image resolution
   - Adjust enhancement parameters
   - Verify license plate region detection

### Debug Mode

```bash
# Enable verbose output
python inference.py -i examples -o results --enhance_plates --verbose

# Test detection only
python test_license_plate.py
```

## Customization

### **Custom YOLO Models**

```python
# Load custom license plate detection model
model = YOLO('path/to/custom_license_plate_model.pt')
```

### **Custom Enhancement Pipeline**

```python
def custom_plate_enhancement(plate_region):
    # Add your custom enhancement logic here
    enhanced = apply_custom_filters(plate_region)
    return enhanced
```

### **Region-Specific Detection**

```python
# Adjust parameters for specific regions/countries
if region == "US":
    aspect_ratio_range = [2.5, 5.0]
elif region == "EU":
    aspect_ratio_range = [2.0, 4.5]
```

## Examples

### **Before Enhancement**
- Hazy license plate with poor contrast
- Blurred text and numbers
- Low visibility in poor lighting

### **After Enhancement**
- Clear, high-contrast license plate
- Sharp, readable text
- Improved visibility in various conditions

## Integration with Existing Workflow

The license plate detection feature integrates seamlessly with the existing IPC-Dehaze pipeline:

1. **Image Upload** → Same as before
2. **Dehazing** → Standard IPC-Dehaze processing
3. **License Plate Detection** → New feature (optional)
4. **Plate Enhancement** → Automatic enhancement of detected regions
5. **Output** → Dehazed image with enhanced license plates

## Future Enhancements

- **Multi-language OCR**: Support for different license plate formats
- **Real-time processing**: Optimized for video streams
- **Custom training**: Tools for training on specific license plate types
- **API integration**: REST API for batch processing
- **Cloud deployment**: Scalable cloud-based processing

## Support

For issues related to license plate detection:

1. Check the troubleshooting section above
2. Run the test script: `python test_license_plate.py`
3. Verify all dependencies are installed
4. Check image quality and format requirements

## License

This feature is part of the IPC-Dehaze project and follows the same license terms.
