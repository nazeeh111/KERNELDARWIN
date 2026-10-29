/** Exact line comparison with bounded alignment memory. Source is never evaluated. */
export function lineDiff(before = "", after = "", cellBudget = 500000) {
  const left = before.split("\n"),
    right = after.split("\n");
  let prefix = 0,
    suffix = 0;
  while (
    prefix < left.length &&
    prefix < right.length &&
    left[prefix] === right[prefix]
  )
    prefix++;
  while (
    suffix < left.length - prefix &&
    suffix < right.length - prefix &&
    left[left.length - 1 - suffix] === right[right.length - 1 - suffix]
  )
    suffix++;
  const a = left.slice(prefix, left.length - suffix),
    b = right.slice(prefix, right.length - suffix);
  const rows = [];
  let oldLine = 0,
    newLine = 0;
  const push = (kind, text) =>
    rows.push({
      kind,
      text,
      old: kind === "add" ? null : ++oldLine,
      new: kind === "remove" ? null : ++newLine,
    });
  for (let i = 0; i < prefix; i++) push("same", left[i]);
  const bounded = (a.length + 1) * (b.length + 1) > cellBudget;
  if (bounded) {
    for (const line of a) push("remove", line);
    for (const line of b) push("add", line);
  } else {
    const width = b.length + 1;
    const lengths = new Uint32Array((a.length + 1) * width);
    for (let i = a.length - 1; i >= 0; i--)
      for (let j = b.length - 1; j >= 0; j--)
        lengths[i * width + j] =
          a[i] === b[j]
            ? 1 + lengths[(i + 1) * width + j + 1]
            : Math.max(
                lengths[(i + 1) * width + j],
                lengths[i * width + j + 1],
              );
    let i = 0,
      j = 0;
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j])
        (push("same", a[i++]), j++);
      else if (
        i < a.length &&
        (j === b.length ||
          lengths[(i + 1) * width + j] >= lengths[i * width + j + 1])
      )
        push("remove", a[i++]);
      else push("add", b[j++]);
    }
  }
  for (let i = left.length - suffix; i < left.length; i++)
    push("same", left[i]);
  return {
    rows,
    bounded,
    additions: rows.filter((r) => r.kind === "add").length,
    removals: rows.filter((r) => r.kind === "remove").length,
  };
}
