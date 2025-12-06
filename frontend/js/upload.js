// Upload Page Logic

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingSpinner = document.getElementById('loading-spinner');

let selectedFile = null;

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
});

// Analyze Button Click
analyzeBtn.addEventListener('click', async function () {
    if (!selectedFile) return;

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

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const result = await response.json();

        // Save result to local storage to pass to results page
        localStorage.setItem('analysisResult', JSON.stringify(result));

        // Redirect
        window.location.href = 'results.html';

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
        loadingSpinner.style.display = 'none';
        analyzeBtn.disabled = false;
        clearBtn.disabled = false;
    }
});
