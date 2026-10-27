class IPCDehazeApp {
    constructor() {
        this.uploadedFiles = [];
        this.actualUploadedFiles = []; // Store actual File objects
        this.sessionId = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkSession();
    }

    bindEvents() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        document.getElementById('processBtn').addEventListener('click', this.processImages.bind(this));
        document.getElementById('clearBtn').addEventListener('click', this.clearAll.bind(this));
        document.getElementById('downloadAllBtn').addEventListener('click', this.downloadAll.bind(this));
        document.getElementById('newSessionBtn').addEventListener('click', this.newSession.bind(this));

    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.processFiles(files);
    }

    async processFiles(files) {
        const imageFiles = files.filter(file => file.type.startsWith('image/'));

        if (imageFiles.length === 0) {
            this.showAlert('Please select valid image files', 'error');
            return;
        }

        if (imageFiles.length > 10) {
            this.showAlert('Maximum 10 images allowed', 'error');
            return;
        }

        const formData = new FormData();
        imageFiles.forEach(file => formData.append('files', file));

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.sessionId = result.session_id;
                this.uploadedFiles = result.files;
                // Store the actual File objects for canvas loading
                this.actualUploadedFiles = imageFiles;
                console.log('Stored actual files:', this.actualUploadedFiles);
                console.log('Stored file info:', this.uploadedFiles);
                this.displayFiles();
                this.showAlert(result.message, 'success');
            } else {
                this.showAlert(result.error, 'error');
            }
        } catch (error) {
            this.showAlert('Upload failed: ' + error.message, 'error');
        }
    }

    displayFiles() {
        const fileGrid = document.getElementById('fileGrid');
        const uploadedFiles = document.getElementById('uploadedFiles');
        const autoDetectionInfo = document.getElementById('autoDetectionInfo');

        fileGrid.innerHTML = '';

        this.uploadedFiles.forEach(file => {
            const fileCard = document.createElement('div');
            fileCard.className = 'file-card';

            // Create image preview using FileReader
            const preview = this.createImagePreview(file);

            fileCard.innerHTML = `
                ${preview}
                <div class="file-info">
                    <div class="file-name">${file.filename}</div>
                    <div class="file-details">
                        ${file.width && file.height ? `${file.width} × ${file.height}px<br>` : ''}
                        ${file.size_mb} MB
                    </div>
                </div>
            `;

            // Add click-to-enlarge on the preview image
            setTimeout(() => {
                const imgEl = fileCard.querySelector('img[data-filename="' + file.filename.replace(/"/g, '&quot;') + '"]');
                if (imgEl) {
                    imgEl.addEventListener('click', () => this.openImageModal(imgEl.src));
                    imgEl.style.cursor = 'zoom-in';
                    imgEl.title = 'Click to view full size';
                }
            }, 0);

            fileGrid.appendChild(fileCard);
        });

        uploadedFiles.style.display = 'block';
        autoDetectionInfo.style.display = 'block';

        // Show processing options immediately
        document.getElementById('processingOptions').style.display = 'block';
    }

    createImagePreview(file) {
        // Try to create a preview using the uploaded File object if available
        const actual = this.actualUploadedFiles.find(f => f.name === file.filename);
        if (actual) {
            const url = URL.createObjectURL(actual);
            return `<div class="file-preview"><img data-filename="${file.filename}" src="${url}" alt="${file.filename}"></div>`;
        }
        // Fallback placeholder
        return `<div class="file-preview">
            <i class="fas fa-image"></i>
            <div style="font-size: 0.8rem; margin-top: 5px;">${file.filename}</div>
        </div>`;
    }

    openImageModal(src) {
        const overlay = document.getElementById('imageModal');
        const img = document.getElementById('imageModalImg');
        img.src = src;
        overlay.style.display = 'flex';
    }

    closeImageModal() {
        const overlay = document.getElementById('imageModal');
        const img = document.getElementById('imageModalImg');
        img.src = '';
        overlay.style.display = 'none';
    }

    async processImages() {
        const processBtn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');

        processBtn.disabled = true;
        loading.style.display = 'block';
        progressBar.style.display = 'block';

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, 500);

        const params = {
            num_iterations: parseInt(document.getElementById('numIterations').value),
            max_size: parseInt(document.getElementById('maxSize').value),
            resize_to: parseInt(document.getElementById('resizeTo').value),
            force_width: parseInt(document.getElementById('forceWidth').value),
            force_height: parseInt(document.getElementById('forceHeight').value),
            enhance_plates: true  // Always true for automatic detection
        };

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const result = await response.json();

            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (result.success) {
                setTimeout(() => {
                    loading.style.display = 'none';
                    this.displayResults(result);
                    this.showAlert(result.message, 'success');
                }, 1000);
            } else {
                loading.style.display = 'none';
                this.showAlert(result.error, 'error');
            }
        } catch (error) {
            clearInterval(progressInterval);
            loading.style.display = 'none';
            this.showAlert('Processing failed: ' + error.message, 'error');
        } finally {
            processBtn.disabled = false;
        }
    }

    async processImages() {
        const btn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');

        btn.disabled = true;
        loading.style.display = 'block';
        progressBar.style.display = 'block';

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 20;
            if (progress > 95) progress = 95;
            progressFill.style.width = progress + '%';
        }, 500);

        const params = {
            num_iterations: parseInt(document.getElementById('numIterations').value),
            max_size: parseInt(document.getElementById('maxSize').value),
            resize_to: parseInt(document.getElementById('resizeTo').value),
            force_width: parseInt(document.getElementById('forceWidth').value),
            force_height: parseInt(document.getElementById('forceHeight').value)
        };

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            const result = await response.json();

            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (result.success) {
                setTimeout(() => {
                    loading.style.display = 'none';
                    this.displayResults(result);
                    this.showAlert(result.message, 'success');
                }, 1000);
            } else {
                loading.style.display = 'none';
                this.showAlert(result.error, 'error');
            }
        } catch (error) {
            clearInterval(progressInterval);
            loading.style.display = 'none';
            this.showAlert('Processing failed: ' + error.message, 'error');
        } finally {
            btn.disabled = false;
        }
    }


    displayResults(apiResult) {
        const resultsCard = document.getElementById('resultsCard');
        const statsBar = document.getElementById('statsBar');

        // Display processed results
        const processedResults = document.getElementById('processedResults');
        const dehazedResults = document.getElementById('dehazedResults');
        const plateCrops = document.getElementById('plateCrops');
        const detectionInfo = document.getElementById('detectionInfo');

        const results = apiResult.results || [];
        const dehazed = apiResult.dehazed_results || [];
        const crops = apiResult.plate_crops || [];

        // Dehazed images
        if (dehazed.length > 0) {
            let dhtml = '';
            dehazed.forEach((item, index) => {
                dhtml += `
                    <div style="margin-bottom: 20px;">
                        <img src="/download/${this.sessionId}/${item.filename}" class="result-image" alt="Dehazed Image ${index + 1}" style="max-width: 100%; height: auto;">
                        <div style="margin-top: 10px;">
                            <a href="/download/${this.sessionId}/${item.filename}" class="result-download">
                                <i class="fas fa-download"></i> Download Dehazed Image ${index + 1}
                            </a>
                        </div>
                    </div>
                `;
            });
            dehazedResults.innerHTML = dhtml;
        } else {
            dehazedResults.innerHTML = '<p style="color:#6b7280">No dehazed images found</p>';
        }

        // Processed results
        if (results.length > 0) {
            let resultsHTML = '';
            results.forEach((result, index) => {
                resultsHTML += `
                    <div style="margin-bottom: 20px;">
                        <img src="/download/${this.sessionId}/${result.filename}" class="result-image" alt="Processed Image ${index + 1}" style="max-width: 100%; height: auto;">
                        <div style="margin-top: 10px;">
                            <a href="/download/${this.sessionId}/${result.filename}" class="result-download">
                                <i class="fas fa-download"></i> Download Processed Image ${index + 1}
                            </a>
                        </div>
                    </div>
                `;
            });
            processedResults.innerHTML = resultsHTML;
        } else {
            processedResults.innerHTML = '<p style="color:#6b7280">No processed images found</p>';
        }

        // Plate crops
        if (crops.length > 0) {
            let chtml = '';
            crops.forEach((item, index) => {
                chtml += `
                    <div style="margin-bottom: 10px;">
                        <img src="/download/${this.sessionId}/${item.filename}" class="result-image" alt="Plate Crop ${index + 1}" style="max-width: 100%; height: auto;">
                    </div>
                `;
            });
            plateCrops.innerHTML = chtml;
        } else {
            plateCrops.innerHTML = '<p style="color:#6b7280">No plate crops found</p>';
        }

        // Display detection information
        const infoList = apiResult.detection_info || [];
        if (infoList.length > 0) {
            let detectionHTML = '';
            infoList.forEach(info => {
                detectionHTML += `<div class="detection-item"><i class="fas fa-check-circle"></i> ${info}</div>`;
            });
            detectionInfo.innerHTML = detectionHTML;
        } else {
            detectionInfo.innerHTML = '<div class="detection-item"><i class="fas fa-info-circle"></i> No license plates detected</div>';
        }

        // Update stats
        document.getElementById('totalImages').textContent = results.length + dehazed.length;
        document.getElementById('processedImages').textContent = results.length;
        let totalSize = results.reduce((sum, r) => sum + r.size_mb, 0) + dehazed.reduce((s, r) => s + r.size_mb, 0);
        document.getElementById('totalSize').textContent = totalSize.toFixed(1);

        statsBar.style.display = 'block';
        resultsCard.style.display = 'block';
    }


    async downloadAll() {
        // This would need to be implemented as a zip download on the backend
        this.showAlert('Download all feature coming soon!', 'success');
    }

    async newSession() {
        try {
            await fetch('/cleanup', { method: 'POST' });
            this.clearAll();
            this.showAlert('New session started', 'success');
        } catch (error) {
            this.showAlert('Failed to start new session', 'error');
        }
    }

    clearAll() {
        this.uploadedFiles = [];
        this.actualUploadedFiles = [];
        this.sessionId = null;

        // Clear canvas
        if (this.canvas && this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.currentImage = null;
            this.selections = [];
        }

        document.getElementById('uploadedFiles').style.display = 'none';
        document.getElementById('imageSelection').style.display = 'none';
        document.getElementById('processingOptions').style.display = 'none';
        document.getElementById('resultsCard').style.display = 'none';
        document.getElementById('statsBar').style.display = 'none';
        document.getElementById('fileInput').value = '';

        this.showAlert('All files cleared', 'success');
    }

    async checkSession() {
        try {
            const response = await fetch('/status');
            const status = await response.json();

            if (status.has_files) {
                this.sessionId = status.session_id;
                // Could restore session state here
            }
        } catch (error) {
            console.log('Session check failed:', error);
        }
    }

    showAlert(message, type) {
        const alert = document.getElementById('alert');
        alert.textContent = message;
        alert.className = `alert alert-${type}`;
        alert.style.display = 'block';

        setTimeout(() => {
            alert.style.display = 'none';
        }, 5000);
    }

    // Selection tool methods
    initSelectionTool() {
        this.canvas = document.getElementById('selectionCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.selections = [];
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentImage = null;

        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.endDrawing.bind(this));
    }

    startDrawing(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.startX = e.clientX - rect.left;
        this.startY = e.clientY - rect.top;
        this.isDrawing = true;
    }

    draw(e) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        // Clear canvas and redraw image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.currentImage) {
            this.ctx.drawImage(this.currentImage, 0, 0, this.canvas.width, this.canvas.height);
        }

        // Draw existing selections
        this.drawSelections();

        // Draw current selection
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(this.startX, this.startY, currentX - this.startX, currentY - this.startY);
    }

    endDrawing(e) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;

        // Create selection box
        const selection = {
            x: Math.min(this.startX, endX),
            y: Math.min(this.startY, endY),
            width: Math.abs(endX - this.startX),
            height: Math.abs(endY - this.startY)
        };

        // Only add if selection is large enough
        if (selection.width > 20 && selection.height > 20) {
            this.selections.push(selection);
            this.updateSelectionStatus();
            this.drawSelections();
        }

        this.isDrawing = false;
    }

    drawSelections() {
        this.selections.forEach((selection, index) => {
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([]);
            this.ctx.strokeRect(selection.x, selection.y, selection.width, selection.height);

            // Add selection number
            this.ctx.fillStyle = '#00ff00';
            this.ctx.font = '16px Arial';
            this.ctx.fillText(`${index + 1}`, selection.x + 5, selection.y + 20);
        });
    }

    clearSelection() {
        this.selections = [];
        this.updateSelectionStatus();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.currentImage) {
            this.ctx.drawImage(this.currentImage, 0, 0, this.canvas.width, this.canvas.height);
        }
    }

    updateSelectionStatus() {
        const statusText = document.querySelector('.status-text');
        const selectionCount = document.querySelector('.selection-count');

        if (this.selections.length === 0) {
            statusText.textContent = 'No regions selected';
            selectionCount.textContent = '0 regions';
        } else {
            statusText.textContent = 'Regions selected';
            selectionCount.textContent = `${this.selections.length} region${this.selections.length > 1 ? 's' : ''}`;
        }
    }

    confirmSelection() {
        if (this.selections.length === 0) {
            this.showAlert('Please select at least one region first', 'error');
            return;
        }

        // Hide selection tool and show processing options
        document.getElementById('imageSelection').style.display = 'none';
        document.getElementById('processingOptions').style.display = 'block';

        this.showAlert(`Confirmed ${this.selections.length} selection(s)`, 'success');
    }

    loadImageToCanvas(imageFile) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                // Calculate canvas size to fit image with better aspect ratio handling
                const maxWidth = 800;
                const maxHeight = 600;
                let { width, height } = img;

                // Calculate aspect ratios
                const imageAspect = width / height;
                const canvasAspect = maxWidth / maxHeight;

                let canvasWidth, canvasHeight;

                if (imageAspect > canvasAspect) {
                    // Image is wider than canvas
                    canvasWidth = maxWidth;
                    canvasHeight = maxWidth / imageAspect;
                } else {
                    // Image is taller than canvas
                    canvasHeight = maxHeight;
                    canvasWidth = maxHeight * imageAspect;
                }

                // Store original image dimensions for coordinate scaling
                this.originalImageWidth = width;
                this.originalImageHeight = height;
                this.canvasDisplayWidth = canvasWidth;
                this.canvasDisplayHeight = canvasHeight;

                this.canvas.width = canvasWidth;
                this.canvas.height = canvasHeight;

                // Draw image
                this.ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
                this.currentImage = img;

                // Show success message briefly
                this.ctx.fillStyle = '#10b981';
                this.ctx.font = '14px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText('Image loaded! Draw selection boxes around license plates', canvasWidth / 2, canvasHeight - 20);

                // Clear the message after 3 seconds
                setTimeout(() => {
                    if (this.currentImage === img) {
                        this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);
                        this.ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
                    }
                }, 3000);

                resolve();
            };
            img.src = URL.createObjectURL(imageFile);
        });
    }

    loadImageToCanvasFromFile(fileInfo) {
        console.log('Loading image:', fileInfo);
        console.log('Available files:', this.actualUploadedFiles);

        // Show loading state
        const canvas = document.getElementById('selectionCanvas');
        const ctx = canvas.getContext('2d');

        // Clear canvas and show loading message
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#6b7280';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Loading image...', canvas.width / 2, canvas.height / 2);

        // Get the actual file from the stored files
        const actualFile = this.actualUploadedFiles.find(f => f.name === fileInfo.filename);
        if (actualFile) {
            console.log('Found actual file:', actualFile);
            this.loadImageToCanvas(actualFile);
        } else {
            // Fallback: try to load from the file info
            console.log('Loading image from file info:', fileInfo);
            console.log('File not found in actualUploadedFiles');
            // We need to store the actual File objects when uploading
            this.showAlert('Image loading issue - please try uploading again', 'error');
        }
    }
}

// Welcome modal functions
function closeWelcomeModal() {
    document.getElementById('welcomeModal').style.display = 'none';
    // Store in localStorage so it doesn't show again in this session
    localStorage.setItem('welcomeShown', 'true');
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Show welcome modal only if it hasn't been shown before
    if (!localStorage.getItem('welcomeShown')) {
        document.getElementById('welcomeModal').style.display = 'flex';
    } else {
        document.getElementById('welcomeModal').style.display = 'none';
    }

    window.app = new IPCDehazeApp();
});
