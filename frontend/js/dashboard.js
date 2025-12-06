// Dashboard Logic

document.addEventListener('DOMContentLoaded', function () {
    fetchMetrics();
});

async function fetchMetrics() {
    try {
        const response = await fetch('http://localhost:5000/metrics');
        if (!response.ok) throw new Error('Failed to fetch metrics');

        const data = await response.json();
        updateMetrics(data);
        renderCharts(data);
        updateHistoryTable(data.predictions); // Using predictions count as mock history for now

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

function updateHistoryTable(predictions) {
    // Mock history data since the backend /metrics endpoint doesn't return full history list in this demo
    // In a real app, we would fetch a list of recent scans
    const tbody = document.getElementById('history-table-body');
    tbody.innerHTML = '';

    // Create some dummy rows for demonstration if no real history is passed
    const dummyData = [
        { id: 'Scan-001', result: 'Tumor Detected', confidence: '98.5%', date: '2025-12-03' },
        { id: 'Scan-002', result: 'No Tumor', confidence: '99.1%', date: '2025-12-03' },
        { id: 'Scan-003', result: 'No Tumor', confidence: '94.2%', date: '2025-12-02' }
    ];

    dummyData.forEach(row => {
        const tr = document.createElement('tr');
        const isTumor = row.result.includes('Tumor Detected');
        tr.innerHTML = `
            <td>${row.id}</td>
            <td><span class="badge ${isTumor ? 'bg-danger' : 'bg-success'}">${row.result}</span></td>
            <td>${row.confidence}</td>
            <td>${row.date}</td>
        `;
        tbody.appendChild(tr);
    });
}
