// Sample presets for quick testing
const SAMPLES = [
  "This movie was an absolute masterpiece! The acting and direction were phenomenal.",
  "Completely waste of time and money. I walked out before the interval.",
  "Not bad at all, in fact it was one of the finest gadgets I've ever owned!",
  "I was expecting a lot, but this product failed on every single metric and broke in two days."
];

let sessionHistory = [];

document.addEventListener("DOMContentLoaded", () => {
  // Check backend device info
  fetchDeviceInfo();

  // Load history from session storage if exists
  const savedHistory = sessionStorage.getItem("bert_history");
  if (savedHistory) {
    try {
      sessionHistory = JSON.parse(savedHistory);
      renderHistory();
    } catch (e) {
      console.error("Failed to parse history", e);
    }
  }

  // Keyboard shortcut: Ctrl + Enter or Cmd + Enter
  const textarea = document.getElementById("reviewInput");
  textarea.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      analyzeSentiment();
    }
  });
});

// Fetch system & GPU info from API
async function fetchDeviceInfo() {
  try {
    const res = await fetch("/api/info");
    if (res.ok) {
      const data = await res.json();
      const pill = document.getElementById("deviceText");
      if (data.device_name) {
        pill.textContent = `${data.device_name} (${data.device.toUpperCase()})`;
      }
    }
  } catch (err) {
    console.warn("Could not reach /api/info", err);
  }
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  if (tabId === "classifier") {
    document.getElementById("tabClassifier").classList.add("active");
    document.getElementById("viewClassifier").classList.add("active");
  } else if (tabId === "metrics") {
    document.getElementById("tabMetrics").classList.add("active");
    document.getElementById("viewMetrics").classList.add("active");
  }
}

// Preset loader
function loadSample(index) {
  const textarea = document.getElementById("reviewInput");
  textarea.value = SAMPLES[index];
  handleInput(textarea);
  analyzeSentiment();
}

// Character counter
function handleInput(elem) {
  const counter = document.getElementById("charCounter");
  const count = elem.value.length;
  counter.textContent = `${count} character${count === 1 ? '' : 's'}`;
}

// Clear textarea
function clearInput() {
  const textarea = document.getElementById("reviewInput");
  textarea.value = "";
  handleInput(textarea);
  textarea.focus();
}

// Analyze Sentiment via API
async function analyzeSentiment() {
  const textarea = document.getElementById("reviewInput");
  const text = textarea.value.trim();

  if (!text) {
    textarea.focus();
    return;
  }

  const btn = document.getElementById("analyzeBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnText = document.getElementById("btnText");
  const latencyBadge = document.getElementById("latencyBadge");

  // Set loading state
  btn.disabled = true;
  btnSpinner.style.display = "inline-block";
  btnText.textContent = "Classifying...";
  latencyBadge.textContent = "Processing...";

  const startTime = performance.now();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error(`Server returned status: ${response.status}`);
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);

    latencyBadge.textContent = `${data.latency_ms || elapsed} ms`;

    // Render results
    renderPredictionResult(data);

    // Add to history
    addToHistory({
      text: text,
      sentiment: data.sentiment,
      confidence: data.confidence,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    });

  } catch (error) {
    console.error("Inference Error:", error);
    latencyBadge.textContent = "Error";
    alert("Inference request failed. Please check if the backend server is running.");
  } finally {
    btn.disabled = false;
    btnSpinner.style.display = "none";
    btnText.textContent = "Analyze with BERT";
  }
}

// Render Prediction into Result Card
function renderPredictionResult(data) {
  document.getElementById("emptyState").classList.add("hidden");
  const details = document.getElementById("resultDetails");
  details.classList.remove("hidden");

  const isPositive = data.sentiment.toLowerCase() === "positive";

  // Banner
  const banner = document.getElementById("sentimentBanner");
  const tag = document.getElementById("sentimentTag");
  const emoji = document.getElementById("sentimentEmoji");
  const confValue = document.getElementById("confidenceValue");

  banner.className = `sentiment-banner ${isPositive ? 'banner-positive' : 'banner-negative'}`;
  tag.className = `sentiment-tag ${isPositive ? 'positive-text' : 'negative-text'}`;
  tag.textContent = data.sentiment.toUpperCase();
  emoji.textContent = isPositive ? "😊" : "😞";
  confValue.textContent = `${(data.confidence * 100).toFixed(2)}%`;

  // Probability bars
  const posProb = (data.prob_positive * 100).toFixed(1);
  const negProb = (data.prob_negative * 100).toFixed(1);

  document.getElementById("posProbText").textContent = `${posProb}%`;
  document.getElementById("posProgressBar").style.width = `${posProb}%`;

  document.getElementById("negProbText").textContent = `${negProb}%`;
  document.getElementById("negProgressBar").style.width = `${negProb}%`;

  // Token inspector
  const pillContainer = document.getElementById("tokenPillContainer");
  const countBadge = document.getElementById("tokenCountBadge");
  pillContainer.innerHTML = "";

  if (data.tokens && data.tokens.length > 0) {
    countBadge.textContent = `${data.tokens.length} subword tokens`;

    data.tokens.forEach((token, idx) => {
      const pill = document.createElement("span");
      let pillClass = "token-pill";

      if (token === "[CLS]" || token === "[SEP]" || token === "[PAD]") {
        pillClass += " token-special";
      } else if (token.startsWith("##")) {
        pillClass += " token-subword";
      }

      pill.className = pillClass;
      const tokenId = data.token_ids ? data.token_ids[idx] : "";
      pill.innerHTML = `<span>${token}</span><span class="token-id">#${tokenId}</span>`;
      pill.title = `Token: ${token} | Vocabulary ID: ${tokenId}`;
      pillContainer.appendChild(pill);
    });
  } else {
    countBadge.textContent = "";
  }
}

// Add prediction to history
function addToHistory(entry) {
  sessionHistory.unshift(entry);
  if (sessionHistory.length > 10) sessionHistory.pop(); // keep last 10
  sessionStorage.setItem("bert_history", JSON.stringify(sessionHistory));
  renderHistory();
}

function renderHistory() {
  const container = document.getElementById("historyList");
  if (!sessionHistory || sessionHistory.length === 0) {
    container.innerHTML = `<div class="empty-history">No predictions run yet in this session.</div>`;
    return;
  }

  container.innerHTML = sessionHistory.map(item => `
    <div class="history-item">
      <div class="history-text" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</div>
      <span class="history-badge ${item.sentiment.toLowerCase() === 'positive' ? 'pos' : 'neg'}">
        ${item.sentiment} (${(item.confidence * 100).toFixed(0)}%)
      </span>
    </div>
  `).join("");
}

function clearHistory() {
  sessionHistory = [];
  sessionStorage.removeItem("bert_history");
  renderHistory();
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, m => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  })[m]);
}
