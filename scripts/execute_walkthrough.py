"""Execute every walkthrough cell in a fresh kernel and reject incomplete results."""
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient

source = Path(__file__).resolve().parents[1] / "notebooks/usage_walkthrough.ipynb"
notebook = nbformat.read(source, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
with tempfile.TemporaryDirectory(prefix="walkthrough-execution-") as directory:
    NotebookClient(
        notebook, timeout=600, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": directory}},
    ).execute()
cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
assert cells, "No runnable examples"
assert all(cell.execution_count is not None for cell in cells), "Unexecuted cell"
assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
output = source.parents[1] / "artifacts/usage_walkthrough.executed.ipynb"
output.parent.mkdir(exist_ok=True)
nbformat.write(notebook, output)
print(f"Executed {len(cells)} code cells: {output}")
