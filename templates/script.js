document.getElementById('uploadBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];

    if (file) {
        // Process the file (add your logic for OCR and summarization here)
        // For now, just simulate output
        document.getElementById('ocrText').innerText = "Sample OCR text extracted from the file.";
        document.getElementById('summaryText').innerText = "This is a summary of the extracted text.";
    } else {
        alert("Please select a file to upload.");
    }
});
