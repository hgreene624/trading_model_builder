#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys, textwrap
from pathlib import Path

PINS = {
    # Stable set that plays nice with older Streamlit UI usage
    "streamlit": "1.32.2",
    "pandas": "2.1.4",
    "numpy": "1.26.4",
    "pyarrow": "14.0.2",
    "plotly": "5.19.0",
}

def sh(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)

def backup_streamlit_config(go: bool):
    home = Path.home()
    cfg_dir = home / ".streamlit"
    if cfg_dir.exists():
        dst = home / f".streamlit.bak"
        if go:
            if dst.exists(): shutil.rmtree(dst, ignore_errors=True)
            shutil.move(str(cfg_dir), str(dst))
            print(f"moved {cfg_dir} -> {dst}")
        else:
            print(f"(dry-run) would move {cfg_dir} -> {dst}")
    else:
        print("~/.streamlit not present; nothing to back up.")

def nuke_and_make_venv(go: bool):
    venv = Path(".venv")
    if go and venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
        print("removed .venv")
    else:
        print("(dry-run) would remove .venv")

    if go:
        sh("python3.11 -m venv .venv")
        print("created .venv")
        act = "source .venv/bin/activate && "
        sh(act + "python -m pip install -U pip setuptools wheel")
    else:
        print("(dry-run) would create .venv and upgrade pip/setuptools/wheel")

def install_pins(go: bool):
    act = "source .venv/bin/activate && "
    pins_line = " ".join(f'{k}=={v}' for k,v in PINS.items())
    if go:
        sh(act + f"python -m pip install {pins_line}")
        # Install the rest of project deps AFTER pins (if you have requirements.txt)
        if Path("requirements.txt").exists():
            sh(act + "python -m pip install -r requirements.txt")
    else:
        print(f"(dry-run) would pip install {pins_line}")
        if Path("requirements.txt").exists():
            print("(dry-run) would pip install -r requirements.txt")

def print_versions(go: bool):
    act = "source .venv/bin/activate && "
    cmd = (
        "python - <<'PY'\n"
        "import sys\n"
        "mods=['streamlit','pandas','numpy','pyarrow','plotly']\n"
        "print('python', sys.version.split()[0])\n"
        "for m in mods:\n"
        "  try:\n"
        "    mod=__import__(m); print(m, getattr(mod,'__version__','unknown'))\n"
        "  except Exception as e:\n"
        "    print(m, 'NOT INSTALLED', e)\n"
        "print('exec', sys.executable)\n"
        "PY\n"
    )
    if go: sh(act + cmd)
    else: print("(dry-run) would print versions inside .venv")

def write_smoke(go: bool):
    code = textwrap.dedent("""\
        import streamlit as st, pandas as pd
        st.title("Smoke")
        df = pd.DataFrame({"a":[1,2,3]})
        st.dataframe(df, height=200)  # no width flags
        st.button("OK")
        st.slider("N", 0, 10, 5)
        st.success("UI primitives render.")
    """)
    p = Path("smoke_app.py")
    if go:
        p.write_text(code)
        print(f"wrote {p}")
    else:
        print(f"(dry-run) would write {p}")

def main():
    ap = argparse.ArgumentParser(description="Deterministic env rebuild for Streamlit app.")
    ap.add_argument("--go", action="store_true", help="execute actions (not just dry-run)")
    args = ap.parse_args()

    print("=== Backup user Streamlit config (if any) ===")
    backup_streamlit_config(args.go)

    print("\n=== Recreate .venv cleanly ===")
    nuke_and_make_venv(args.go)

    print("\n=== Install pinned versions ===")
    install_pins(args.go)

    print("\n=== Verify versions ===")
    print_versions(args.go)

    print("\n=== Write smoke_app.py ===")
    write_smoke(args.go)

    print("\nNext steps:")
    print("  1) source .venv/bin/activate")
    print("  2) streamlit run smoke_app.py  # should render OK")
    print("  3) streamlit run Home.py       # re-test your app")
    print("If things still break, paste the FIRST traceback after this rebuild.")

if __name__ == "__main__":
    main()
