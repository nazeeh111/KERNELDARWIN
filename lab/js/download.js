/** Export only explicit text supplied by the user or derived from a record. */
export function downloadText(text, filename, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  // Prevent imported identifiers from becoming path separators or control bytes.
  link.download = filename.replace(/[\x00-\x1f/\\:*?"<>|]/g, "_").slice(0, 180);
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
