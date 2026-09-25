// chain review page: stress-chart tooltip; click a step on the chart to jump to its row
document.querySelectorAll(".chart").forEach((fig) => {
  const tip = fig.querySelector(".tip");
  fig.querySelectorAll(".hit").forEach((hit) => {
    hit.addEventListener("mousemove", (e) => {
      const box = fig.getBoundingClientRect();
      tip.hidden = false;
      tip.textContent = hit.dataset.tip;
      tip.style.left = Math.min(e.clientX - box.left + 12, box.width - 260) + "px";
      tip.style.top = e.clientY - box.top + 12 + "px";
    });
    hit.addEventListener("mouseleave", () => (tip.hidden = true));
    hit.addEventListener("click", () => (location.hash = "step-" + hit.dataset.step));
  });
});
