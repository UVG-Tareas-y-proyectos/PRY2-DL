"""Execute every notebook cell and retain visible outputs in the committed file."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/proyecto2.ipynb"
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
notebook = nbformat.read(path, as_version=4)
executed = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                          resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(executed, path)
print("Executed", path)
print("Code cells with output", sum(bool(cell.get("outputs")) for cell in executed.cells
                                    if cell.cell_type == "code"))
