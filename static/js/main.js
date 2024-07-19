function fetchClassCounts() {
    fetch('/class_counts')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('detection-list');
            list.innerHTML = '';
            for (const [cls, count] of Object.entries(data.counts)) {
                const listItem = document.createElement('li');
                listItem.innerText = `${cls}: ${count}`;
                list.appendChild(listItem);
            }
        })
        .catch(error => console.error('Error fetching class counts:', error));
}

function updateSettings() {
    const confidence = document.getElementById('confidence-slider').value;
    const iouThreshold = document.getElementById('iou-slider').value;
    const modelId = document.getElementById('model-id').value;
    
    fetch('/update_parameters', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            confidence: parseFloat(confidence),
            iou_threshold: parseFloat(iouThreshold),
            model_id: modelId,
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Settings updated successfully');
        } else {
            console.error('Error updating settings');
        }
    })
    .catch(error => console.error('Error updating settings:', error));
}

// Fetch the class counts every second
setInterval(fetchClassCounts, 1000);

// Update the slider values dynamically
document.getElementById('confidence-slider').addEventListener('input', function() {
    document.getElementById('confidence-value').innerText = this.value;
});

document.getElementById('iou-slider').addEventListener('input', function() {
    document.getElementById('iou-value').innerText = this.value;
});

// Update settings on button click
document.getElementById('update-settings').addEventListener('click', updateSettings);