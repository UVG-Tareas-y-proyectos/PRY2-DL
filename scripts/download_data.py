"""Fetch the public IBM HI-Small CSV mirror without redistributing raw data."""
from pathlib import Path
from urllib.request import urlretrieve

target = Path(__file__).resolve().parents[1] / "data/raw/HI-Small_Trans.csv"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    url = "https://huggingface.co/datasets/OsamaMIT/IBM-AML-HI-Small/resolve/main/HI-Small_Trans.csv?download=true"
    urlretrieve(url, target)
if target.stat().st_size < 400_000_000:
    raise RuntimeError("The CSV appears incomplete; remove it and retry")
print(target, target.stat().st_size)
