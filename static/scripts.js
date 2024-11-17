// Start the slice animation and transition to the next page after form submission
function uploadFile(event) {
  event.preventDefault(); // Prevent the form from submitting normally

  // Start the slice animation
  startSliceAnimation();

  // Grab the form data
  const form = event.target;
  const formData = new FormData(form);

  // Add form data to localStorage or pass it to the backend (in this case, localStorage)
  const formDataObject = {};
  formData.forEach((value, key) => {
    formDataObject[key] = value;
  });

  // Store data temporarily in localStorage (you can replace this with a backend POST request)
  localStorage.setItem("formData", JSON.stringify(formDataObject));

  // Simulate upload and transition to graphs page after animation completes
  setTimeout(() => {
    window.location.href = "graphs.html"; // Navigate to the next page after animation
  }, 800); // Ensure this matches the animation duration
}

// Start the slice animation
function startSliceAnimation() {
  const overlay = document.getElementById("slice-overlay");

  // Activate the slice animation
  overlay.classList.add("active");
}
