// Display the selected file name
function showFileName() {
  const fileInput = document.getElementById("csv-file");
  const fileName = document.getElementById("file-name");
  if (fileInput.files.length > 0) {
    fileName.textContent = `Selected File: ${fileInput.files[0].name}`;
  } else {
    fileName.textContent = "";
  }
}

// Start the slice animation and transition to the next page
function uploadFile(event) {
  event.preventDefault(); // Prevent the form from submitting

  // Start the slice animation
  startSliceAnimation();

  // Simulate upload success
  fetch("/upload", {
    method: "POST",
    body: new FormData(event.target), // Send the form data
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data.message); // File upload successful
      // You can add more logic here to handle the data if needed
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function startSliceAnimation() {
  const overlay = document.getElementById("slice-overlay");

  // Activate the slice animation
  overlay.classList.add("active");

  // Delay page transition until after animation completes (0.8s)
  setTimeout(() => {
    window.location.href = "graphs.html"; // Navigate to the next page after animation
  }, 800); // Ensure this matches the animation duration
}
