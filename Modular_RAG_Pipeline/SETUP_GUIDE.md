# Conda Environment & RAG Setup Guide

This guide documents the exact setup steps we performed to configure the isolated virtual environment (`rag_env`), install the correct dependencies, and resolve VS Code interpreter and linting issues. Refer to this guide if you need to recreate the environment or fix python paths in the future.

---

## 1. Environment Activation & Base Setup

To ensure you are using the correct isolated environment, open your terminal inside the project directory and run:

```bash
# 1. Activate your Conda environment
conda activate rag_env

# 2. (Optional) If recreating the environment from scratch, install Python 3.10 and Pip inside it
conda install python=3.10 pip -y
```

---

## 2. Installing Project Dependencies

Because macOS can sometimes route the default `pip` command to the system's global Python (e.g., Python 3.14), **always use `python -m pip`** when installing packages. This forces the system to install them inside the active Conda environment.

Run this command from your root directory:

```bash
# Install all required packages for LangChain and PDF processing
python -m pip install -r RAG/requirements.txt

# Install ipykernel so Jupyter Notebook can communicate with this environment
python -m pip install ipykernel
```

> [!TIP]
> **Best Practice:** Always run `python -m pip install <package>` instead of `pip install <package>`. This guarantees the packages are installed into the active Conda environment, avoiding conflicts with the macOS system Python.

---

## 3. Configuring VS Code Settings

To prevent VS Code's terminal from inheriting global system environment paths (which causes the terminal to use system-wide `pip`), we created a workspace settings file at:
📂 [settings.json](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/.vscode/settings.json)

It contains:
```json
{
    "terminal.integrated.inheritEnv": false
}
```
* **Effect:** When you open a fresh terminal, it will start cleanly, allowing Conda to successfully override Python paths upon activation.

---

## 4. Connecting the Environment to your Jupyter Notebook

When writing code in `.ipynb` files (like [document.ipynb](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/NoteBook/document.ipynb)):

1. Open your `.ipynb` notebook file.
2. In the **top-right corner**, click on the current kernel selection (e.g. `Select Kernel` or `Python 3.14...`).
3. Select **Select Another Kernel...** $\rightarrow$ **Python Environments**.
4. Choose the environment labeled:
   👉 **`Python 3.10.20 ('rag_env')`** (located at `/opt/homebrew/Caskroom/miniforge/...`)

---

## 5. Syncing VS Code Linting (Removing Red Squiggly Lines)

If your code runs successfully but VS Code displays red squiggly lines under your imports (like `import langchain_core`), you need to tell VS Code's code analyzer (Pylance) which interpreter to check:

1. Open the Command Palette:
   * **macOS:** `Cmd + Shift + P`
   * **Windows/Linux:** `Ctrl + Shift + P`
2. Search for: **`Python: Select Interpreter`** and select it.
3. Click on **`Python 3.10.20 ('rag_env')`**.
4. (Optional) If the red lines don't disappear immediately, open the Command Palette again, search for **`Developer: Reload Window`**, and run it to restart the code analyzer.

---

## 6. Environment Verification Cheat-Sheet

Run these commands inside your activated environment terminal to verify everything is set up correctly:

| Command | Expected Correct Output | What it means |
| :--- | :--- | :--- |
| `which python` | `/opt/homebrew/Caskroom/miniforge/base/envs/rag_env/bin/python` | Python is running from the isolated environment. |
| `python -m pip --version` | `pip ... from .../envs/rag_env/lib/... (python 3.10)` | Pip is installing packages into the isolated environment. |

Within a Jupyter Notebook cell, you can also run:
```python
import sys
print(sys.executable)
# Should print: /opt/homebrew/Caskroom/miniforge/base/envs/rag_env/bin/python
```
