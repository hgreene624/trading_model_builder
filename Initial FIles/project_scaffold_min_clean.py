#!/usr/bin/env python3
"""
Minimal scaffolder for the Streamlit trading app.

This ONLY creates the directories and EMPTY placeholder files so you can fill them later.
It will NOT write implementation code or your secrets.

Usage:
  python project_scaffold_min_clean.py [target_dir] [--overwrite]

If omitted, target_dir defaults to ./streamlit-trader
"""
import sys
from pathlib import Path

PLACEHOLDER_LINE = "# TODO: implement\n"

DIRS = [
    ".",
    "pages",
    "src",
    "src/models",
    "src/data",
    "src/utils",
    "storage",
    ".streamlit",
]

FILES = [
    "Home.py",
    "pages/2_Ticker_Selector_and_Tuning.py",
    "pages/3_Portfolios.py",
    "pages/4_Simulate_Portfolio.py",
    "src/__init__.py",
    "src/storage.py",
    "src/models/__init__.py",
    "src/data/__init__.py",
    "src/utils/__init__.py",
    "src/utils/plotting.py",
    "src/data/alpaca_data.py",
    "src/data/yf.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".env",
    ".streamlit/config.toml",
    ".streamlit/secrets.toml",
    "storage/.gitkeep",
]

CONTENT = {
    "Home.py": "# TODO: Dashboard page\n",
    "pages/2_Ticker_Selector_and_Tuning.py": "# TODO: Ticker selector & tuning page\n",
    "pages/3_Portfolios.py": "# TODO: Portfolios management page\n",
    "pages/4_Simulate_Portfolio.py": "# TODO: Portfolio simulation page\n",
    "src/__init__.py": "",
    "src/storage.py": "# TODO: JSON storage helpers\n",
    "src/models/__init__.py": "",
    "src/data/__init__.py": "",
    "src/utils/__init__.py": "",
    "src/utils/plotting.py": "# TODO: Plotting helpers (e.g., equity curve)\n",
    "src/data/alpaca_data.py": "# TODO: Alpaca OHLCV loader\n",
    "src/data/yf.py": "# TODO: yfinance fallback loader (optional)\n",
    "requirements.txt": "# TODO: streamlit, alpaca-py, pandas, plotly, etc.\n",
    "README.md": "# TODO: Project README\n",
    ".gitignore": ".env\n.streamlit/secrets.toml\nstorage/\n__pycache__/\n*.pyc\n.DS_Store\n",
    ".env": "# ALPACA_API_KEY=\n# ALPACA_SECRET_KEY=\n# ALPACA_BASE_URL=https://paper-api.alpaca.markets\n# ALPACA_DATA_URL=https://data.alpaca.markets\n",
    ".streamlit/config.toml": "# Optional Streamlit theme/config\n",
    ".streamlit/secrets.toml": "# Streamlit secrets (DO NOT COMMIT)\n# ALPACA_API_KEY = \"...\"\n# ALPACA_SECRET_KEY = \"...\"\n",
    "storage/.gitkeep": "",
}

def write_text(path: Path, text: str, overwrite: bool=False):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True

def main():
    # Parse args
    args = sys.argv[1:]
    overwrite = False
    target = None
    for a in args:
        if a == "--overwrite":
            overwrite = True
        elif target is None:
            target = a
    if target is None:
        target = "streamlit-trader"

    root = Path(target).resolve()

    # Make dirs
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Make files
    for f in FILES:
        p = root / f
        text = CONTENT.get(f, PLACEHOLDER_LINE)
        created = write_text(p, text, overwrite=overwrite)
        action = "created" if created else "skipped"
        print(f"{action:8} {p.relative_to(root)}")  # simple log

    print("\n✔ Scaffolding complete at:", str(root))
    print("Next:")
    print("  cd", str(root))
    print("  # Fill in the placeholder files, then:")
    print("  # python -m venv .venv && source .venv/bin/activate")
    print("  # pip install -r requirements.txt")
    print("  # streamlit run Home.py")

if __name__ == "__main__":
    main()
