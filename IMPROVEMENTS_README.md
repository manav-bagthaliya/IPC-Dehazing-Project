# IPC-Dehaze Improvements

This document outlines the comprehensive improvements made to the IPC-Dehaze project to address quality, visibility, and sizing issues.

## 🎯 Issues Addressed

### 1. **Full Dehazed Image Quality**
- **Problem**: Dehazed images were not clear enough
- **Solution**: Added advanced post-processing with CLAHE, contrast enhancement, and bilateral filtering

### 2. **License Plate Number Visibility**
- **Problem**: Numbers on license plates were unclear in selected regions
- **Solution**: Implemented multi-step enhancement pipeline specifically for license plate regions

### 3. **Downloaded Image Size**
- **Problem**: Downloaded images were too small
- **Solution**: Improved coordinate scaling and maintained original image dimensions

## 🔧 Technical Improvements

### Enhanced License Plate Detection & Enhancement

**File**: `inference.py` - `enhance_plate_region()` function

**Multi-step Enhancement Pipeline**:
1. **LAB Color Space Conversion**: Better contrast enhancement
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Adaptive contrast improvement
3. **Strong Contrast Enhancement**: Alpha=2.5, Beta=60 for maximum visibility
4. **Unsharp Masking**: Advanced sharpening technique
5. **Bilateral Filtering**: Noise reduction while preserving edges
6. **Morphological Operations**: Text character enhancement
7. **Additional Sharpening**: Kernel-based sharpening for text clarity
8. **Final Contrast Boost**: Alpha=1.4, Beta=30 for readability

```python
# Example of the enhanced pipeline
lab = cv2.cvtColor(plate_region, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])
enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
enhanced = cv2.convertScaleAbs(enhanced, alpha=2.5, beta=60)
# ... additional steps
```

### Improved Output Quality Enhancement

**File**: `inference.py` - `enhance_output_quality()` function

**Quality Enhancement Steps**:
1. **LAB Color Space Processing**: Better color enhancement
2. **CLAHE Application**: Adaptive contrast improvement
3. **Subtle Contrast Enhancement**: Alpha=1.1, Beta=10
4. **Bilateral Filtering**: Noise reduction with edge preservation
5. **Value Clipping**: Ensures valid pixel values

### Better Coordinate Scaling

**Files**: `app.py` and `templates/index.html`

**Improvements**:
- **Aspect Ratio Preservation**: Proper canvas sizing based on image dimensions
- **Accurate Coordinate Mapping**: Frontend sends actual canvas dimensions to backend
- **Original Size Preservation**: Maintains original image dimensions in output

```javascript
// Frontend: Better canvas sizing
const imageAspect = width / height;
const canvasAspect = maxWidth / maxHeight;

if (imageAspect > canvasAspect) {
    canvasWidth = maxWidth;
    canvasHeight = maxWidth / imageAspect;
} else {
    canvasHeight = maxHeight;
    canvasWidth = maxHeight * imageAspect;
}
```

### High-Quality Image Saving

**Files**: `inference.py` and `app.py`

**Improvements**:
- **PNG Compression Level 0**: Maximum quality output
- **Original Dimension Preservation**: No unnecessary resizing
- **Quality Enhancement**: Post-processing for better results

```python
# High-quality saving
cv2.imwrite(output_path, output_img_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])
```

## 📁 File Changes Summary

### Backend Changes (`app.py`)
- ✅ Enhanced `enhance_selected_region()` function with advanced processing
- ✅ Improved `process_selected_area()` with better coordinate scaling
- ✅ Added canvas dimension parameters to processing pipeline
- ✅ High-quality image saving with PNG compression level 0

### Inference Changes (`inference.py`)
- ✅ Enhanced `enhance_plate_region()` with 10-step processing pipeline
- ✅ Added `enhance_output_quality()` function for better output
- ✅ Improved resizing logic to preserve original dimensions
- ✅ Better quality preservation throughout processing

### Frontend Changes (`templates/index.html`)
- ✅ Better canvas sizing with aspect ratio preservation
- ✅ Improved coordinate scaling with actual canvas dimensions
- ✅ Enhanced image loading with proper dimension tracking
- ✅ Better selection tool with accurate coordinate mapping

## 🧪 Testing

### Test Script: `test_improvements.py`

Run the test suite to verify improvements:

```bash
python test_improvements.py
```

**Tests Include**:
- ✅ License plate enhancement quality
- ✅ Output quality enhancement
- ✅ Coordinate scaling accuracy
- ✅ Image dimension preservation

## 🚀 Usage Instructions

### 1. **Start the Application**
```bash
cd "D:\7 SEM\IPC-Dehaze"
venv\Scripts\activate
python app.py
```

### 2. **Upload Images**
- Open browser: `http://localhost:5000`
- Upload images with license plates
- Images will be displayed with proper sizing

### 3. **Select License Plate Regions**
- Click on uploaded images to load them in the selection tool
- Draw selection boxes around license plates
- Multiple selections are supported

### 4. **Process Images**
- Configure processing options:
  - **Number of Iterations**: 8-20 (higher = better quality, slower)
  - **Max Size**: 1500px (larger images scaled down)
  - **Enhance License Plates**: Check for automatic enhancement
- Click "Process Images"

### 5. **Download Results**
- **Full Image Dehazed**: Complete dehazed image with enhanced quality
- **Selected Areas Enhanced**: Individual license plate regions with maximum visibility
- All images maintain original dimensions and high quality

## 📊 Quality Improvements

### Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **License Plate Visibility** | Unclear numbers | Crystal clear text |
| **Image Quality** | Basic dehazing | Advanced enhancement |
| **File Size** | Small downloads | Original dimensions |
| **Coordinate Accuracy** | Approximate scaling | Precise mapping |

### Enhancement Pipeline

1. **Input Processing**: Proper aspect ratio handling
2. **Dehazing**: Advanced AI-based fog removal
3. **License Plate Detection**: YOLO + OpenCV fallback
4. **Region Enhancement**: 10-step processing pipeline
5. **Quality Enhancement**: Post-processing for clarity
6. **High-Quality Output**: PNG compression level 0

## 🔍 Technical Details

### License Plate Enhancement Steps

1. **LAB Color Space**: Better contrast processing
2. **CLAHE**: Adaptive histogram equalization
3. **Contrast Boost**: Alpha=2.5, Beta=60
4. **Unsharp Masking**: Advanced sharpening
5. **Bilateral Filter**: Noise reduction
6. **Morphological Operations**: Text enhancement
7. **Sharpening Kernel**: Text clarity
8. **Final Contrast**: Alpha=1.4, Beta=30
9. **Value Clipping**: Valid pixel range
10. **OCR Integration**: Optional text recognition

### Coordinate Scaling Logic

```python
# Accurate scaling from canvas to original image
if original_width > 0 and original_height > 0:
    scale_x = original_width / canvas_width
    scale_y = original_height / canvas_height
else:
    # Fallback to calculated dimensions
    image_aspect = img_width / img_height
    canvas_aspect = canvas_width / canvas_height
    # ... aspect ratio calculations
```

## 🎉 Results

### Expected Improvements

1. **✅ Clearer Full Images**: Advanced dehazing with quality enhancement
2. **✅ Visible License Plate Numbers**: Multi-step enhancement pipeline
3. **✅ Proper Image Sizes**: Original dimensions preserved
4. **✅ High-Quality Downloads**: PNG compression level 0
5. **✅ Accurate Selections**: Precise coordinate scaling

### Performance Notes

- **Processing Time**: Slightly increased due to enhanced quality
- **Memory Usage**: Minimal increase for quality processing
- **Output Quality**: Significantly improved visibility and clarity
- **File Sizes**: Maintained at original dimensions

## 🛠️ Troubleshooting

### Common Issues

1. **Images Still Unclear**
   - Increase "Number of Iterations" to 12-16
   - Check "Enhance License Plates" option
   - Ensure good quality input images

2. **Selection Not Accurate**
   - Refresh the page and try again
   - Ensure proper image loading in canvas
   - Check browser console for errors

3. **Download Issues**
   - Check file permissions in results directory
   - Ensure sufficient disk space
   - Try different browser if issues persist

### Debug Information

Enable debug mode in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

Check console output for processing details and any errors.

## 📈 Future Enhancements

### Potential Improvements

1. **AI-Powered License Plate Detection**: Custom trained model
2. **Real-time Processing**: WebSocket-based live updates
3. **Batch Processing**: Multiple image enhancement
4. **Quality Metrics**: PSNR, SSIM measurements
5. **Custom Enhancement**: User-adjustable parameters

### Performance Optimizations

1. **GPU Acceleration**: CUDA support for faster processing
2. **Memory Optimization**: Efficient tensor operations
3. **Parallel Processing**: Multi-threaded enhancement
4. **Caching**: Result caching for repeated operations

---

**Note**: These improvements significantly enhance the quality and usability of the IPC-Dehaze system, providing clear, high-quality results with visible license plate numbers and proper image sizing.
