// the form elements from the page
const imageForm = document.getElementById('imageForm');
const textForm = document.getElementById('textForm');
const videoForm = document.getElementById('videoForm');
// elements used to display the results
const resultArea = document.getElementById('resultArea');
const resultImg = document.getElementById('resultImage');
const imageResultContainer = document.getElementById('imageResultContainer');
const videoResultContainer = document.getElementById('videoResultContainer');
const resultVideo = document.getElementById('resultVideo');
const resultVideoSource = document.getElementById('resultVideoSource');
// download and loading elements
const downloadLink = document.getElementById('downloadLink');
const loading = document.getElementById('loading');

// helper function to handle form submission
async function handleFormSubmit(event, endpoint, isVideo = false) {
    // stop the form from refreshing the page
    event.preventDefault();
    // get the form that was submitted
    const form = event.currentTarget;
    const formData = new FormData(form);
    // hide old results and show the loading message
    resultArea.style.display = 'none';
    imageResultContainer.style.display = 'none';
    videoResultContainer.style.display = 'none';
    loading.style.display = 'block';
    // pausing the video player if active
    resultVideo.pause();
    resultVideoSource.src = "";
    resultVideo.load();

    try {
        // send the form data to the server
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });
        // checking if the server returned an error
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }
        // get the server response as a file
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        if (isVideo) {
            // load the output video
            resultVideoSource.src = url;
            resultVideo.load();
            // show the video result and alink for the download
            videoResultContainer.style.display = 'block';
            downloadLink.download = 'stylized_video.mp4';
        } 
        else {
            // load output image
            resultImg.src = url;
            imageResultContainer.style.display = 'block';
            downloadLink.download = 'stylized_image.jpg';
        }
        // set the result url on the download link
        downloadLink.href = url;
        resultArea.style.display = 'block';
    } 
    // show an error message if the request fails
    catch (err) {
        alert(`Error: ${err.message}`);
    } 
    finally {
        loading.style.display = 'none';
    }
}

// the event listeners
// submit event listener to the image form
if (imageForm) {
    imageForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-image', false));
}
// submit event listener to the text form
if (textForm) {
    textForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-text', false));
}
// add a submit event listener to the video form
if (videoForm) {
    videoForm.addEventListener('submit', (e) => handleFormSubmit(e, '/style-transfer-video', true));
}
