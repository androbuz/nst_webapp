// DOM elements
const imageForm = document.getElementById('imageForm');
const textForm = document.getElementById('textForm');
const videoForm = document.getElementById('videoForm');

const resultArea = document.getElementById('resultArea');
const resultImg = document.getElementById('resultImage');
const imageResultContainer = document.getElementById('imageResultContainer');

const videoResultContainer = document.getElementById('videoResultContainer');
const resultVideo = document.getElementById('resultVideo');
const resultVideoSource = document.getElementById('resultVideoSource');

const downloadLink = document.getElementById('downloadLink');
const loading = document.getElementById('loading');

// helper function to handle form submission
async function handleFormSubmit(event, endpoint, isVideo = false) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);

    // Hide previous results, show loading
    resultArea.style.display = 'none';
    imageResultContainer.style.display = 'none';
    videoResultContainer.style.display = 'none';
    loading.style.display = 'block';

    // Pause video player if active
    resultVideo.pause();
    resultVideoSource.src = "";
    resultVideo.load();

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

        if (isVideo) {
            // Load output video source
            resultVideoSource.src = url;
            resultVideo.load();
            videoResultContainer.style.display = 'block';
            downloadLink.download = 'stylized_video.mp4';
        } else {
            // Load output image
            resultImg.src = url;
            imageResultContainer.style.display = 'block';
            downloadLink.download = 'stylized_image.jpg';
        }

        downloadLink.href = url;
        resultArea.style.display = 'block';
    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

// Attach event listeners
if (imageForm) {
    imageForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-image', false));
}
if (textForm) {
    textForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-text', false));
}
if (videoForm) {
    videoForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-video', true));
}
