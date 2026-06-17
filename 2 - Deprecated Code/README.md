## Quickstart

Install via PyPI standard practices:
```bash
deactivate 2>/dev/null || true
rm -rf venv
pyenv install -s 3.12.13
pyenv local 3.12.13
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```
or
```bash
pip install -e ".[cuda12]"
```

To redo
```bash
deactivate
rm -rf venv
```

THIS CODEBASE IS A GOOD TEMPLATE BUT ITS MATH IS NO LONGER RELIABLE.