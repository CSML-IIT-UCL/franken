"use strict";

function setupAgentOnboarding() {
  document.querySelectorAll(".agent-onboarding__copy").forEach((button) => {
    if (button.dataset.ready) return;
    button.dataset.ready = "true";
    const guide = new URL(button.dataset.agentGuide, document.baseURI).href;
    const prompt = `Install Franken for this project by following the instructions at ${guide}.`;
    const label = button.querySelector("[data-copy-label]");
    let resetTimer;

    button.addEventListener("click", async () => {
      clearTimeout(resetTimer);
      try {
        await navigator.clipboard.writeText(prompt);
        label.textContent = "Prompt copied!";
        resetTimer = setTimeout(() => {
          label.textContent = "Copy setup prompt";
        }, 2500);
      } catch {
        label.textContent = "Copy setup prompt";
        window.prompt("Copy this prompt into your agent:", prompt);
      }
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setupAgentOnboarding);
} else {
  setupAgentOnboarding();
}
