// Public dataset repository.
const DATASET_URL = "https://huggingface.co/datasets/s1m0neee/TUTUS";
const datasetLink = document.getElementById("dataset-link");
if (datasetLink && /^https:\/\//.test(DATASET_URL)) {
  datasetLink.href = DATASET_URL;
  datasetLink.removeAttribute("aria-disabled");
  datasetLink.removeAttribute("tabindex");
  datasetLink.classList.remove("is-disabled");
  datasetLink.title = "Download TUTUS";
}
const copyButton = document.getElementById("copy-bibtex");
if (copyButton) copyButton.addEventListener("click", async () => {
  const text = document.getElementById("citation").textContent;
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
    status.textContent = "Citation copied to clipboard.";
    setTimeout(() => { copyButton.textContent = "Copy citation"; }, 2000);
  } catch (_) {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById("citation"));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = "Select and copy the highlighted citation.";
  }
});
