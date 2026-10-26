/** Short, interruptible orientation transitions. No timers drive recorded data. */
const preference = matchMedia("(prefers-reduced-motion: reduce)");
const active = new Set();
preference.addEventListener("change", (event) => {
  if (event.matches) for (const animation of active) animation.cancel();
});
export function reveal(node, subtle = false) {
  for (const animation of active) animation.cancel();
  active.clear();
  if (preference.matches || !node.animate) return;
  const animation = node.animate(
    [
      { opacity: 0.35, transform: `translateY(${subtle ? 2 : 7}px)` },
      { opacity: 1, transform: "translateY(0)" },
    ],
    { duration: subtle ? 150 : 240, easing: "cubic-bezier(.2,.75,.25,1)" },
  );
  active.add(animation);
  animation.finished.catch(() => {}).finally(() => active.delete(animation));
}
