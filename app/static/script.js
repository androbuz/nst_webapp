// Get DOM elements
const imageForm = document.getElementById('imageForm');
const textForm = document.getElementById('textForm');
const resultArea = document.getElementById('resultArea');
const resultImg = document.getElementById('resultImage');
const downloadLink = document.getElementById('downloadLink');
const loading = document.getElementById('loading');

// Helper: handle form submission
async function handleFormSubmit(event, endpoint) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);

    // Hide previous result, show loading
    resultArea.style.display = 'none';
    loading.style.display = 'block';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        resultImg.src = url;
        downloadLink.href = url;
        resultArea.style.display = 'block';
    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

// Attach event listeners
imageForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-image'));
textForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-text'));