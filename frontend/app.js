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
  statusText.textContent = isBusy ? "Generating answer..." : "Ready";
  statusText.classList.toggle("busy", isBusy);
  
  if (isBusy) {
    statusText.classList.add("typing");
  } else {
    statusText.classList.remove("typing");
  }
}

function appendMessage(role, text, sources = [], variant = role.toLowerCase()) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(variant);

  node.querySelector(".role").textContent = role;
  node.querySelector(".text").textContent = text;
  node.querySelector(".stamp").textContent = formatTime();

  const sourcesEl = node.querySelector(".sources");
  if (!sources || sources.length === 0) {
    sourcesEl.remove();
  } else {
    sourcesEl.textContent = "Sources:";
    const ul = document.createElement("ul");
    for (const src of sources) {
      const li = document.createElement("li");
      li.textContent = src;
      ul.appendChild(li);
    }
    sourcesEl.appendChild(ul);
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function setupChips() {
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      promptInput.value = chip.dataset.prompt || chip.innerText.trim();
      form.requestSubmit();
    });
  });
}

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const query = promptInput.value.trim();
  if (!query) return;

  promptInput.value = "";
  promptInput.style.height = "auto";
  
  appendMessage("You", query);
  setBusyState(true);

  try {
    const res = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query }),
    });

    if (!res.ok) throw new Error("Server response was not ok");
    
    const data = await res.json();
    appendMessage("Assistant", data.message, data.sources);
  } catch (err) {
    console.error("Chat error:", err);
    appendMessage("System", "Failed to connect to the assistant server. Please ensure the backend API is running.", [], "error");
  } finally {
    setBusyState(false);
    promptInput.focus();
  }
});

promptInput.addEventListener("input", function() {
  this.style.height = "auto";
  this.style.height = (this.scrollHeight) + "px";
});

setupChips();
