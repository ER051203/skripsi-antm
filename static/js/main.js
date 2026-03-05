// static/js/main.js

// Initialize tooltips
document.addEventListener("DOMContentLoaded", function () {
  // Auto-hide alerts after 5 seconds
  setTimeout(function () {
    let alerts = document.querySelectorAll(".alert");
    alerts.forEach(function (alert) {
      alert.style.transition = "opacity 0.5s ease";
      alert.style.opacity = "0";
      setTimeout(function () {
        alert.remove();
      }, 500);
    });
  }, 5000);

  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });

  // Format currency inputs
  let currencyInputs = document.querySelectorAll(".currency-input");
  currencyInputs.forEach((input) => {
    input.addEventListener("input", function (e) {
      let value = this.value.replace(/[^0-9]/g, "");
      if (value) {
        this.value = new Intl.NumberFormat("id-ID").format(value);
      }
    });
  });
});

// Loading state untuk form submission
document.querySelectorAll("form").forEach((form) => {
  form.addEventListener("submit", function (e) {
    let submitBtn = this.querySelector('button[type="submit"]');
    if (submitBtn) {
      let originalText = submitBtn.innerHTML;
      submitBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin me-2"></i>Memproses...';
      submitBtn.disabled = true;

      // Re-enable after timeout (just in case)
      setTimeout(function () {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
      }, 30000);
    }
  });
});

// Live stock price update
function updateStockPrice(ticker = "ANTM.JK") {
  fetch(`/api/stock-data?ticker=${ticker}&days=1`)
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.prices.length > 0) {
        let latestPrice = data.prices[data.prices.length - 1];
        let priceElement = document.getElementById("latest-price");
        if (priceElement) {
          priceElement.innerHTML = `Rp ${new Intl.NumberFormat("id-ID").format(latestPrice)}`;

          // Add animation
          priceElement.classList.add("price-update");
          setTimeout(() => {
            priceElement.classList.remove("price-update");
          }, 500);
        }
      }
    })
    .catch((error) => console.error("Error fetching stock price:", error));
}

// Update price every 60 seconds
if (document.getElementById("latest-price")) {
  updateStockPrice();
  setInterval(updateStockPrice, 60000);
}

// Chart interactivity
function exportChartAsImage(chartId, filename) {
  let chartElement = document.getElementById(chartId);
  if (chartElement && chartElement._fullLayout) {
    Plotly.downloadImage(chartElement, {
      format: "png",
      width: 1200,
      height: 800,
      filename: filename || "chart",
    });
  }
}

// Add export button to charts
document.querySelectorAll(".graph-card").forEach((card, index) => {
  let chartId = card.querySelector('div[id^="price-chart"]')?.id;
  if (chartId) {
    let exportBtn = document.createElement("button");
    exportBtn.className = "btn btn-sm btn-export mt-2";
    exportBtn.innerHTML = '<i class="fas fa-download me-2"></i>Export Chart';
    exportBtn.onclick = () => exportChartAsImage(chartId, `chart-${index + 1}`);
    card.querySelector(".card-body")?.appendChild(exportBtn);
  }
});

// Comparison table sorting
function sortTable(column) {
  let table = document.querySelector(".future-card table");
  if (!table) return;

  let rows = Array.from(table.querySelectorAll("tbody tr"));
  let sortDirection = table.dataset.sortDirection === "asc" ? "desc" : "asc";
  table.dataset.sortDirection = sortDirection;

  rows.sort((a, b) => {
    let aVal = a.cells[column].innerText.replace(/[^0-9]/g, "");
    let bVal = b.cells[column].innerText.replace(/[^0-9]/g, "");

    if (sortDirection === "asc") {
      return aVal - bVal;
    } else {
      return bVal - aVal;
    }
  });

  let tbody = table.querySelector("tbody");
  rows.forEach((row) => tbody.appendChild(row));
}

// Add sort indicators to table headers
document.querySelectorAll(".future-card th").forEach((th, index) => {
  if (index > 0) {
    // Skip date column
    th.style.cursor = "pointer";
    th.addEventListener("click", () => sortTable(index));
    th.title = "Click to sort";
  }
});

// Responsive font sizing
function adjustFontSize() {
  let width = window.innerWidth;
  let metricValues = document.querySelectorAll(".metric-value");

  metricValues.forEach((el) => {
    if (width < 768) {
      el.style.fontSize = "1.2rem";
    } else if (width < 992) {
      el.style.fontSize = "1.5rem";
    } else {
      el.style.fontSize = "1.8rem";
    }
  });
}

window.addEventListener("resize", adjustFontSize);
adjustFontSize();

// Dark mode toggle (optional)
function toggleDarkMode() {
  document.body.classList.toggle("light-mode");
  localStorage.setItem(
    "darkMode",
    document.body.classList.contains("light-mode") ? "light" : "dark",
  );
}

// Check saved preference
if (localStorage.getItem("darkMode") === "light") {
  document.body.classList.add("light-mode");
}

// Keyboard shortcuts
document.addEventListener("keydown", function (e) {
  // Ctrl + P untuk print
  if (e.ctrlKey && e.key === "p") {
    e.preventDefault();
    window.print();
  }

  // Ctrl + H untuk home
  if (e.ctrlKey && e.key === "h") {
    e.preventDefault();
    window.location.href = "/";
  }
});

// Share functionality
function shareResults() {
  if (navigator.share) {
    navigator
      .share({
        title: "Hasil Prediksi Saham ANTM",
        text: "Lihat hasil prediksi saham menggunakan LSTM dan Linear Regression",
        url: window.location.href,
      })
      .catch(console.error);
  } else {
    // Fallback
    navigator.clipboard.writeText(window.location.href);
    alert("URL hasil prediksi telah disalin ke clipboard!");
  }
}

// Add share button
let shareBtn = document.createElement("button");
shareBtn.className = "btn btn-share btn-lg";
shareBtn.innerHTML = '<i class="fas fa-share-alt me-2"></i>Share';
shareBtn.onclick = shareResults;
document.querySelector(".action-buttons")?.appendChild(shareBtn);
