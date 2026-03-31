const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send-btn");
const messagesEl = document.getElementById("messages");
const template = document.getElementById("msg-template");
const statusText = document.getElementById("status-text");
const chips = document.querySelectorAll(".chip");

function formatTime(now = new Date()) {
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function setBusyState(isBusy) {
  sendBtn.disabled = isBusy;
  sendBtn.textContent = isBusy ? "Thinking" : "Send";
  statusText.textContent = isBusy ? "Generating answer" : "Ready";
  statusText.classList.toggle("busy", isBusy);
}

function appendMessage(role, text, sources = [], variant = role.toLowerCase()) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(variant);

  node.querySelector(".role").textContent = role;
  node.querySelector(".text").textContent = text;
  node.querySelector(".stamp").textContent = formatTime();

  const sourcesEl = node.querySelector(".sources");
  if (sources.length === 0) {
    sourcesEl.remove();
  } else {
    for (const src of sources) {
      const li = document.createElement("li");
      li.textContent = src;
      sourcesEl.appendChild(li);
    }
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

appendMessage(
  "Assistant",
  "Hi! I can help with IBA admissions policy, fees, registration, and required documents. Ask anything.",
  []
);

chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    promptInput.value = chip.dataset.prompt || "";
    promptInput.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = promptInput.value.trim();
  if (!message) {
    return;
  }

  appendMessage("You", message);
  promptInput.value = "";
  setBusyState(true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = payload?.detail || "Request failed";
      throw new Error(detail);
    }

    appendMessage("Assistant", payload.answer || "No answer returned.", payload.sources || []);
  } catch (err) {
    appendMessage("Assistant", `Error: ${err.message}`, [], "error");
  } finally {
    setBusyState(false);
    promptInput.focus();
  }
});
