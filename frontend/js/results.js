// Results Page Logic

document.addEventListener('DOMContentLoaded', function () {
    const resultData = localStorage.getItem('analysisResult');

    if (!resultData) {
        alert('No analysis result found. Redirecting to upload page.');
        window.location.href = 'upload.html';
        return;
    }

    const data = JSON.parse(resultData);
    renderResults(data);
});

function renderResults(data) {
    const banner = document.getElementById('result-banner');
    const originalImg = document.getElementById('original-image');
    const processedImg = document.getElementById('processed-image');
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceCircle = document.querySelector('.confidence-circle');

    // Set Images
    // Note: In a real deployment, these paths would need to be absolute or relative to the server root
    originalImg.src = `http://localhost:5000/uploads/${data.filename}`;
    processedImg.src = `http://localhost:5000/results/${data.processed_filename}`;

    // Set Banner
    if (data.is_tumor) {
        banner.className = 'result-banner bg-tumor';
        banner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> TUMOR DETECTED';
    } else {
        banner.className = 'result-banner bg-normal';
        banner.innerHTML = '<i class="fas fa-check-circle"></i> NO TUMOR DETECTED';
    }

    // Set Confidence
    const percentage = (data.confidence * 100).toFixed(1);
    confidenceScore.innerText = `${percentage}%`;

    // Update Circle Gradient
    const degree = data.confidence * 360;
    confidenceCircle.style.background = `conic-gradient(var(--primary-color) ${degree}deg, var(--light-gray) 0deg)`;
}
