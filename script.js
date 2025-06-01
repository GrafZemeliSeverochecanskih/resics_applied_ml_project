// Ensure these elements are selected at the top of your script tag
const imageResultsRow = document.getElementById('imageResultsRow');
const originalImageTag = document.getElementById('originalImageTag');
const saliencyMapImageTag = document.getElementById('saliencyMapImageTag'); // Use new ID if changed

// ... (keep existing selectors like form, imageFile, loader, predictionText, etc.) ...

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    // Clear previous results and errors
    imageResultsRow.style.display = 'none'; // Hide the image row
    originalImageTag.src = '#';
    originalImageTag.style.display = 'none';
    saliencyMapImageTag.src = '#';
    saliencyMapImageTag.style.display = 'none';

    predictionText.textContent = ''; // Clear prediction text
    document.getElementById('predictionDetailsContainer').style.display = 'none'; // Hide text details
    probabilitiesList.innerHTML = '';
    // ... (clear other messages) ...
    errorDisplay.textContent = '';

    if (!imageFile.files || imageFile.files.length === 0) {
        errorDisplay.textContent = 'Please select an image file.';
        return;
    }

    const selectedFile = imageFile.files[0]; // Get the selected file
    const formData = new FormData();
    formData.append('image_file', selectedFile);

    loader.style.display = 'block';

    try {
        const response = await fetch('/predict_and_explain/', { // Assuming this is your target endpoint
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();

            imageResultsRow.style.display = 'flex'; // Show the image row (uses flex from inline style)
            document.getElementById('predictionDetailsContainer').style.display = 'block'; // Show text details

            // Display Original Image
            originalImageTag.src = URL.createObjectURL(selectedFile);
            originalImageTag.style.display = 'block';

            // Display Saliency Map (backend now returns just the heatmap)
            if (data.saliency_map_base64) {
                saliencyMapImageTag.src = `data:image/png;base64,${data.saliency_map_base64}`;
                saliencyMapImageTag.style.display = 'block';
            } else {
                saliencyMapImageTag.style.display = 'none';
                // Optionally, display a message in its container
                document.getElementById('saliencyMapDisplay').innerHTML += '<p>Saliency map not available.</p>';
            }

            // Display prediction text
            predictionText.innerHTML = `<strong>Predicted Class:</strong> ${data.prediction || 'N/A'}`;

            // Display probabilities
            if (data.probabilities && data.probabilities.length > 0) {
                // ... (your existing probability display logic) ...
            } else {
                // ... (handle no probabilities) ...
            }

        } else {
            // ... (your existing error handling) ...
        }
    } catch (error) {
        // ... (your existing error handling) ...
    } finally {
        loader.style.display = 'none';
    }
});