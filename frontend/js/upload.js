// Upload Page Logic

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingSpinner = document.getElementById('loading-spinner');
const diagnosticsPanel = document.getElementById('diagnostics-panel');
const diagnosticsMessage = document.getElementById('diagnostics-message');
const diagnosticsCopyBtn = document.getElementById('diagnostics-copy-btn');

let selectedFile = null;
let diagnosticsShown = false;

function clearDiagnostics() {
    diagnosticsShown = false;
    if (!diagnosticsPanel || !diagnosticsMessage) return;
    diagnosticsMessage.textContent = '';
    diagnosticsPanel.style.display = 'none';
}

function showDiagnostics(message, details) {
    if (!diagnosticsPanel || !diagnosticsMessage) return;
    diagnosticsShown = true;
    const lines = [];
    if (message) lines.push(message);
    if (details) lines.push(details);
    diagnosticsMessage.textContent = lines.join('\n');
    diagnosticsPanel.style.display = 'block';
}

async function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text);
    }
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'absolute';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
}

if (diagnosticsCopyBtn) {
    diagnosticsCopyBtn.addEventListener('click', async function () {
        if (!diagnosticsMessage) return;
        const text = diagnosticsMessage.textContent || '';
        if (!text) return;
        try {
            await copyToClipboard(text);
            const original = diagnosticsCopyBtn.textContent;
            diagnosticsCopyBtn.textContent = 'Copied';
            setTimeout(() => {
                diagnosticsCopyBtn.textContent = original;
            }, 1500);
        } catch (e) {
            // If copy fails, do nothing.
        }
    });
}

// Drag and Drop Events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('dragover');
}

function unhighlight(e) {
    dropZone.classList.remove('dragover');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// File Input Change
fileInput.addEventListener('change', function () {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (validateFile(file)) {
            selectedFile = file;
            showPreview(file);
        }
    }
}

function validateFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload a JPG or PNG image.');
        return false;
    }
    return true;
}

function showPreview(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function () {
        imagePreview.src = reader.result;
        previewSection.style.display = 'block';
        dropZone.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Clear Selection
clearBtn.addEventListener('click', function () {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    dropZone.style.display = 'block';
    analyzeBtn.disabled = true;
    clearDiagnostics();
});

// Analyze Button Click
analyzeBtn.addEventListener('click', async function () {
    if (!selectedFile) return;

    clearDiagnostics();

    // Show Loading
    loadingSpinner.style.display = 'block';
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        let result = null;
        try {
            result = await response.json();
        } catch (e) {
            result = null;
        }

        if (!response.ok) {
            const message = result && result.error ? result.error : `Analysis failed (HTTP ${response.status})`;
            const details = [];
            details.push(`Status: ${response.status} ${response.statusText}`);
            if (result) {
                details.push(`Body: ${JSON.stringify(result, null, 2)}`);
            } else {
                details.push('Body: (no JSON returned)');
            }
            showDiagnostics(message, details.join('\n'));
            throw new Error(message);
        }

        // Save result to local storage to pass to results page
        localStorage.setItem('analysisResult', JSON.stringify(result));

        // Redirect
        window.location.href = 'results.html';

    } catch (error) {
        console.error('Error:', error);
        if (!diagnosticsShown) {
            const message = error && error.message ? error.message : 'An error occurred during analysis.';
            const details = 'Check that the Flask server is running at http://localhost:5000';
            showDiagnostics(message, details);
        }
        alert(error.message || 'An error occurred during analysis. Please try again.');
        loadingSpinner.style.display = 'none';
        analyzeBtn.disabled = false;
        clearBtn.disabled = false;
    }
});
