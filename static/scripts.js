function uploadFile() {
  const fileInput = document.getElementById("csv-file");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select a file!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        alert(data.error);
      } else {
        alert("File uploaded successfully!");
        renderGraphs(data.stats);
      }
    })
    .catch((error) => console.error("Error:", error));
}

function renderGraphs(stats) {
  const graph1 = document.getElementById("graph1").getContext("2d");
  const graph2 = document.getElementById("graph2").getContext("2d");
  const graph3 = document.getElementById("graph3").getContext("2d");

  // Example Graph 1: Fraud Alerts
  new Chart(graph1, {
    type: "bar",
    data: {
      labels: ["Total Transactions", "Potential Fraud"],
      datasets: [
        {
          label: "Counts",
          data: [stats.total_transactions, stats.potential_fraud],
          backgroundColor: ["#0073e6", "#ff5733"],
        },
      ],
    },
  });

  // Example Graph 2: Average Transaction Amount
  new Chart(graph2, {
    type: "pie",
    data: {
      labels: ["Average Amount"],
      datasets: [
        {
          label: "Amount",
          data: [stats.average_amount],
          backgroundColor: ["#28a745"],
        },
      ],
    },
  });

  // Example Graph 3: Dummy Data
  new Chart(graph3, {
    type: "line",
    data: {
      labels: ["1", "2", "3", "4", "5"],
      datasets: [
        {
          label: "Dummy Data",
          data: [10, 20, 30, 40, 50],
          borderColor: "#ffc107",
        },
      ],
    },
  });
}
