"""IPC-Dehaze web interface (Flask).

Run from the project root:  python web/app.py   ->  http://localhost:5000
"""

import json
import os
import shutil
import subprocess
import sys
import uuid

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory, session
from werkzeug.utils import secure_filename

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from ipc_dehaze.config import CRITIC_PATH, DEFAULT_MAX_SIZE, DEFAULT_NUM_ITERATIONS, PREDICTOR_PATH  # noqa: E402
from ipc_dehaze.enhance import binarize_plate  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ipc-dehaze-secret-key-2025')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Security headers
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Configuration
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(ROOT_DIR, 'web_results')
INFERENCE_SCRIPT = os.path.join(ROOT_DIR, 'inference.py')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_IMAGES = 10

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return round(os.path.getsize(file_path) / (1024 * 1024), 2)
    return 0

def get_image_dimensions(file_path):
    """Get image dimensions"""
    try:
        img = cv2.imread(file_path)
        if img is not None:
            height, width = img.shape[:2]
            return width, height
    except Exception as e:
        print(f"Error reading image dimensions: {e}")
    return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')

    if not files or len(files) == 0 or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    if len(files) > MAX_IMAGES:
        return jsonify({'error': f'Maximum {MAX_IMAGES} images allowed'}), 400

    # Validate file types and sizes
    total_size = 0
    for file in files:
        if file and not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type: {file.filename}'}), 400

        # Check individual file size (rough estimate)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        total_size += file_size

        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f'File too large: {file.filename}'}), 400

    if total_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'Total file size exceeds limit'}), 400

    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id

    # Create session directories
    session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
    session_results_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    os.makedirs(session_results_dir, exist_ok=True)

    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_upload_dir, filename)
            file.save(file_path)

            # Get file info
            file_size = get_file_size_mb(file_path)
            width, height = get_image_dimensions(file_path)

            uploaded_files.append({
                'filename': filename,
                'size_mb': file_size,
                'width': width,
                'height': height,
                'path': file_path
            })

    session['uploaded_files'] = uploaded_files
    session['session_upload_dir'] = session_upload_dir
    session['session_results_dir'] = session_results_dir

    return jsonify({
        'success': True,
        'session_id': session_id,
        'files': uploaded_files,
        'message': f'Successfully uploaded {len(uploaded_files)} images'
    })

@app.route('/process', methods=['POST'])
def process_images():
    data = request.get_json()

    if 'session_id' not in session:
        return jsonify({'error': 'No active session'}), 400

    session_upload_dir = session.get('session_upload_dir')
    session_results_dir = session.get('session_results_dir')

    if not session_upload_dir or not session_results_dir:
        return jsonify({'error': 'Session directories not found'}), 400

    # Get processing parameters
    num_iterations = data.get('num_iterations', DEFAULT_NUM_ITERATIONS)
    max_size = data.get('max_size', DEFAULT_MAX_SIZE)
    resize_to = data.get('resize_to', 0)
    force_width = data.get('force_width', 0)
    force_height = data.get('force_height', 0)
    enhance_plates = data.get('enhance_plates', True)  # Default to True for automatic detection

    # Build command for automatic processing (dehaze + plate detection)
    cmd_full = [
        sys.executable or 'python', INFERENCE_SCRIPT,
        '-i', session_upload_dir,
        '-o', session_results_dir,
        '-n', str(num_iterations),
        '--max_size', str(max_size),
        '--enhance_plates'  # Enable automatic plate detection
    ]

    if resize_to > 0:
        cmd_full.extend(['--resize_to', str(resize_to)])

    if force_width > 0 and force_height > 0:
        cmd_full.extend(['--force_width', str(force_width), '--force_height', str(force_height)])

    for model_path in (PREDICTOR_PATH, CRITIC_PATH):
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}. Please ensure the model files are in the pretrained_models folder.'}), 400

    try:
        # Process images with automatic license plate detection
        result_full = subprocess.run(
            cmd_full,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Handle Unicode errors gracefully
            cwd=ROOT_DIR,
            timeout=900  # 15 minute timeout for slower CPUs/large images
        )

        def collect_results():
            result_files = []
            dehazed_files = []
            plate_crops = []
            if os.path.exists(session_results_dir):
                for file in os.listdir(session_results_dir):
                    if file.lower().endswith(('_dehazed.png')):
                        file_path = os.path.join(session_results_dir, file)
                        file_size = get_file_size_mb(file_path)
                        width, height = get_image_dimensions(file_path)
                        dehazed_files.append({
                            'filename': file,
                            'size_mb': file_size,
                            'width': width,
                            'height': height,
                            'path': file_path
                        })
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(session_results_dir, file)
                        file_size = get_file_size_mb(file_path)
                        width, height = get_image_dimensions(file_path)
                        # Plate crops
                        if '_plate_' in file:
                            plate_crops.append({
                                'filename': file,
                                'size_mb': file_size,
                                'width': width,
                                'height': height,
                                'path': file_path
                            })
                            continue
                        # Skip images in subfolders
                        if os.path.isdir(os.path.join(session_results_dir, file)):
                            continue
                        # Rename normal outputs once for clarity
                        if not file.startswith('processed_') and not file.endswith('_dehazed.png'):
                            new_filename = f"processed_{file}"
                            new_file_path = os.path.join(session_results_dir, new_filename)
                            try:
                                os.rename(file_path, new_file_path)
                            except Exception:
                                new_file_path = file_path
                                new_filename = file
                        else:
                            new_file_path = file_path
                            new_filename = file
                        result_files.append({
                            'filename': new_filename,
                            'size_mb': file_size,
                            'width': width,
                            'height': height,
                            'path': new_file_path
                        })
            return result_files, dehazed_files, plate_crops

        if result_full.returncode == 0:
            # Get processed results
            result_files = []
            dehazed_files = []
            plate_crops = []
            detection_info = []
            plates_structured = []

            result_files, dehazed_files, plate_crops = collect_results()

            # Parse detection info from stdout
            stdout_lines = result_full.stdout.split('\n')
            for line in stdout_lines:
                if 'Detected' in line and 'license plate' in line:
                    detection_info.append(line.strip())
                elif 'Automatically detected and enhanced' in line:
                    detection_info.append(line.strip())
                elif 'Recognized plate text:' in line:
                    detection_info.append(line.strip())
                elif line.startswith('PLATE_JSON:'):
                    try:
                        payload = line.split('PLATE_JSON:', 1)[1].strip()
                        plate_obj = json.loads(payload)
                        plates_structured.append(plate_obj)
                        # Human-readable echo
                        if 'formatted_text' in plate_obj and 'confidence' in plate_obj:
                            detection_info.append(f"Plate: {plate_obj['formatted_text']} | Confidence: {plate_obj['confidence']:.2f}")
                    except Exception:
                        pass

            return jsonify({
                'success': True,
                'message': 'Automatic processing completed successfully',
                'results': result_files,
                'dehazed_results': dehazed_files,
                'plate_crops': plate_crops,
                'command': ' '.join(cmd_full),
                'enhanced_plates': enhance_plates,
                'detection_info': detection_info,
                'plates': plates_structured,
                'auto_detection': True
            })
        else:
            # Log to server console for debugging
            print('\n===== Inference Failed =====')
            print('Command:', ' '.join(cmd_full))
            print('Return code:', result_full.returncode)
            print('STDOUT:\n', result_full.stdout[:4000])
            print('STDERR:\n', result_full.stderr[:4000])
            # Fallback: if we still produced any images, return them as partial success
            result_files, dehazed_files, plate_crops = collect_results()
            if result_files or dehazed_files or plate_crops:
                return jsonify({
                    'success': True,
                    'message': 'Processing completed with warnings',
                    'results': result_files,
                    'dehazed_results': dehazed_files,
                    'plate_crops': plate_crops,
                    'command': ' '.join(cmd_full),
                    'enhanced_plates': enhance_plates,
                    'detection_info': ['Warning: inference returned non-zero exit code. See server logs for details.'],
                    'auto_detection': True
                })
            return jsonify({
                'error': 'Processing failed',
                'command': ' '.join(cmd_full),
                'returncode': result_full.returncode,
                'stderr': result_full.stderr,
                'stdout': result_full.stdout
            }), 500

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


def process_selected_area(upload_dir, results_dir, selection, area_number, canvas_width=800, canvas_height=600, original_width=0, original_height=0):
    """Enhance a user-drawn region of the first uploaded image.

    TODO: not wired to any route yet - the UI collects selections but never sends them.
    """
    try:
        # Get the first image from upload directory
        image_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return None

        input_image_path = os.path.join(upload_dir, image_files[0])
        image = cv2.imread(input_image_path)

        if image is None:
            return None

        # Get original image dimensions
        img_height, img_width = image.shape[:2]

        # Extract selected region with better coordinate scaling
        x, y, w, h = selection['x'], selection['y'], selection['width'], selection['height']

        # Use provided canvas dimensions or calculate based on image aspect ratio
        if original_width > 0 and original_height > 0:
            # Use the actual canvas dimensions from frontend
            actual_canvas_width = canvas_width
            actual_canvas_height = canvas_height

            # Scale coordinates from canvas to original image
            scale_x = original_width / actual_canvas_width
            scale_y = original_height / actual_canvas_height
        else:
            # Fallback to calculated dimensions
            image_aspect = img_width / img_height
            canvas_aspect = canvas_width / canvas_height

            if image_aspect > canvas_aspect:
                # Image is wider than canvas
                actual_canvas_width = canvas_width
                actual_canvas_height = canvas_width / image_aspect
            else:
                # Image is taller than canvas
                actual_canvas_height = canvas_height
                actual_canvas_width = canvas_height * image_aspect

            # Scale coordinates properly
            scale_x = img_width / actual_canvas_width
            scale_y = img_height / actual_canvas_height

        # Apply scaling and ensure bounds
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        if w <= 0 or h <= 0:
            return None

        # Extract region
        selected_region = image[y:y+h, x:x+w]

        # Apply enhanced processing to selected region
        enhanced_region = binarize_plate(selected_region)

        # Save enhanced region with high quality
        output_filename = f"selected_enhanced_area_{area_number}.png"
        output_path = os.path.join(results_dir, output_filename)

        # Save with high quality settings
        cv2.imwrite(output_path, enhanced_region, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Get file info
        file_size = get_file_size_mb(output_path)
        width, height = enhanced_region.shape[1], enhanced_region.shape[0]

        return {
            'filename': output_filename,
            'size_mb': file_size,
            'width': width,
            'height': height,
            'path': output_path
        }

    except Exception as e:
        print(f"Error processing selected area: {e}")
        return None

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    if session_id != session.get('session_id'):
        return jsonify({'error': 'Invalid session'}), 400

    session_results_dir = session.get('session_results_dir')
    if not session_results_dir:
        return jsonify({'error': 'Session not found'}), 400

    if not os.path.isfile(os.path.join(session_results_dir, filename)):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(session_results_dir, filename, as_attachment=True)

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    if 'session_id' in session:
        session_id = session['session_id']

        # Clean up directories
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        session_results_dir = os.path.join(RESULTS_FOLDER, session_id)

        if os.path.exists(session_upload_dir):
            shutil.rmtree(session_upload_dir)
        if os.path.exists(session_results_dir):
            shutil.rmtree(session_results_dir)

        # Clear session
        session.clear()

        return jsonify({'success': True, 'message': 'Session cleaned up'})

    return jsonify({'error': 'No active session'}), 400

@app.route('/status')
def get_status():
    if 'session_id' in session:
        return jsonify({
            'session_id': session['session_id'],
            'has_files': 'uploaded_files' in session,
            'file_count': len(session.get('uploaded_files', [])),
            'upload_dir': session.get('session_upload_dir'),
            'results_dir': session.get('session_results_dir')
        })

    return jsonify({'session_id': None, 'has_files': False})

if __name__ == '__main__':
    # Debug mode exposes an interactive debugger; enable only locally via FLASK_DEBUG=1
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
