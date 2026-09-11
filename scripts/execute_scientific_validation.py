"""Execute the source-bundle scientific notebook in a fresh kernel."""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "notebooks/scientific_validation.ipynb", as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
NotebookClient(
    notebook,
    timeout=900,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": str(root)}},
).execute()
for cell in notebook.cells:
    if cell.cell_type == "code" and cell.source.strip() and cell.execution_count is None:
        raise RuntimeError("Scientific validation contains an unexecuted cell")
output = root / "artifacts/scientific_validation.executed.ipynb"
output.parent.mkdir(exist_ok=True)
nbformat.write(notebook, output)
print(f"Every scientific validation cell executed: {output}")
