// Dashboard Logic

document.addEventListener('DOMContentLoaded', function () {
    fetchMetrics();
});

async function fetchMetrics() {
    try {
        const [metricsResponse, historyResponse] = await Promise.all([
            fetch('http://localhost:5000/metrics'),
            fetch('http://localhost:5000/history')
        ]);

        if (!metricsResponse.ok) throw new Error('Failed to fetch metrics');
        if (!historyResponse.ok) throw new Error('Failed to fetch history');

        const metricsData = await metricsResponse.json();
        const historyData = await historyResponse.json();

        updateMetrics(metricsData);
        renderCharts(metricsData);
        updateHistoryTable(historyData);

    } catch (error) {
        console.error('Error:', error);
    }
}

function updateMetrics(data) {
    document.getElementById('metric-accuracy').innerText = (data.accuracy * 100).toFixed(1) + '%';
    document.getElementById('metric-precision').innerText = (data.precision * 100).toFixed(1) + '%';
    document.getElementById('metric-recall').innerText = (data.recall * 100).toFixed(1) + '%';
    document.getElementById('metric-f1').innerText = (data.f1_score * 100).toFixed(1) + '%';
}

function renderCharts(data) {
    // Accuracy Chart
    new Chart(document.getElementById('accuracyChart'), {
        type: 'line',
        data: {
            labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6'],
            datasets: [{
                label: 'Training Accuracy',
                data: data.history.accuracy,
                borderColor: '#0077B6',
                tension: 0.1
            }, {
                label: 'Validation Accuracy',
                data: data.history.val_accuracy,
                borderColor: '#28A745',
                tension: 0.1
            }]
        }
    });

    // Loss Chart
    new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6'],
            datasets: [{
                label: 'Training Loss',
                data: data.history.loss,
                borderColor: '#DC3545',
                tension: 0.1
            }, {
                label: 'Validation Loss',
                data: data.history.val_loss,
                borderColor: '#FFC107',
                tension: 0.1
            }]
        }
    });

    // Pie Chart
    new Chart(document.getElementById('pieChart'), {
        type: 'doughnut',
        data: {
            labels: ['Tumor Detected', 'No Tumor'],
            datasets: [{
                data: [data.predictions.tumor, data.predictions.normal],
                backgroundColor: ['#DC3545', '#28A745']
            }]
        }
    });
}

function updateHistoryTable(historyData) {
    const tbody = document.getElementById('history-table-body');
    tbody.innerHTML = '';

    if (!historyData || historyData.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="4" class="text-center text-muted">No scans yet. Upload an image to get started.</td>';
        tbody.appendChild(tr);
        return;
    }

    historyData.forEach((item, index) => {
        const tr = document.createElement('tr');
        const isTumor = item.is_tumor;
        const scanId = `Scan-${String(historyData.length - index).padStart(3, '0')}`;
        const confidence = (item.confidence * 100).toFixed(1) + '%';
        const date = item.timestamp.split(' ')[0]; // Get date part only

        tr.innerHTML = `
            <td>${scanId}</td>
            <td><span class="badge ${isTumor ? 'bg-danger' : 'bg-success'}">${item.label}</span></td>
            <td>${confidence}</td>
            <td>${date}</td>
        `;
        tbody.appendChild(tr);
    });
}
