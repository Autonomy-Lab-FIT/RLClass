# RLClass
Code for Dr Tiwari's Reinforcement Learning Class

## Install

1. Ensure you have Python **3.11** installed on your Ubuntu 22.04 system. (We have only verified the code works for Python 3.11, but 3.10+ should work fine.)  
   The class README uses Python 3.11 for the `venv` workflow, so this Ubuntu guide keeps the same target version.

   **Ubuntu 22.04 (recommended, no conda):** install Python 3.11 alongside the system Python:
   ```bash
   sudo apt update
   sudo apt install -y software-properties-common git
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install -y python3.11 python3.11-venv python3.11-dev
   python3.11 --version
   ```

   Do **not** replace Ubuntu's system `python3`. Use the versioned `python3.11` command for this repo.

2. **Do not use Conda or Miniconda** for this personal Ubuntu setup.  
   The official class README uses Conda by default, but it already provides a no-conda `venv` workflow for personal setups. This Ubuntu guide follows that same approach.

3. (Optional) We personally prefer using VS Code for our coding editor, but if you are more familiar with something like PyCharm or Sublime you can use that. In order to install VS Code follow the instructions for your operating system here: https://code.visualstudio.com/
4. Launch your code editor. (from here on out, we will assume you are using VS Code.)
5. When you launch VS Code, go to Extensions on the left taskbar and install the following extensions:  
   a. Python  
   b. Python Debugger  
   c. Pylance  
   d. Jupyter  
6. Install Git if you do not already have it:
   ```bash
   sudo apt update
   sudo apt install -y git
   ```
7. Next you will clone the repo from GitHub. Go to https://github.com/Autonomy-Lab-FIT/RLClass and click the green dropdown button that says "<> Code". Copy the HTTPS URL and clone it into your preferred folder, for example:
   ```bash
   mkdir -p ~/Documents/GitHub
   cd ~/Documents/GitHub
   git clone https://github.com/Autonomy-Lab-FIT/RLClass.git
   ```
8. Once you have Git set up and the repo cloned, go into VS Code and open the `RLClass` folder.
9. Open a terminal in VS Code and create your own branch with:
   ```bash
   git checkout -b <YOUR-NAME>-branch
   ```
10. Push your newly created branch to the GitHub repo with:
   ```bash
   git push -u origin <YOUR-NAME>-branch
   ```

---

## Installing Requirements

Once you have created your branch and pushed it to the repo, you are now ready to install the requirements and begin coding.

### Ubuntu 22.04 (no conda): Python `venv` workflow

This section mirrors the MacOS `venv` workflow from the class README, but for Ubuntu 22.04 and VS Code.

1. Open a terminal in VS Code and make sure your directory is pointing to the `RLClass/` folder.
2. Verify Python 3.11 is available:
   ```bash
   python3.11 --version
   ```
3. Create a local virtual environment in the repo:
   ```bash
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
4. Verify the venv is active and using Python 3.11:
   ```bash
   python --version
   command -v python
   ```
   Expected: `python` points to `.../RLClass/.venv/bin/python`.

5. Install requirements:
   ```bash
   python -m pip install --upgrade pip
   ```
   Then install dependencies with the Ubuntu venv requirements file:
   ```bash
   python -m pip install -r requirements.ubuntu.txt
   ```

6. Ensure Jupyter kernel support is installed (VS Code may also prompt for this):
   ```bash
   python -m pip install ipykernel
   ```

7. Open the first Jupyter Notebook. Click **Select Kernel** (top-right) → **Python Environments** → choose the interpreter at `RLClass/.venv/bin/python`.

8. For LunarLander and other Box2D environments, if you run into missing dependency issues, re-run:
   ```bash
   python -m pip install swig
   python -m pip install "gymnasium[box2d]"
   ```

9. Test the environment:
   ```bash
   python -c "import torch, gymnasium; print('torch ok'); print('gymnasium ok')"
   python - <<'PY'
import gymnasium as gym
env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=42)
print("reset ok, obs length:", len(obs))
env.close()
PY
   ```

---

## Ubuntu note: Gym vs Gymnasium

This repo installs both `gym` and `gymnasium`, and the requirements set also includes both packages.  
Recommended for notebooks and new code: use **Gymnasium** everywhere.

```python
import gymnasium as gym
```

and unpack `step()` as:

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

This also matches the newer API style you have already been using in your recent RL work.

---

## VS Code notes

- Use `Ctrl+Shift+P` → **Python: Select Interpreter**
- Select `RLClass/.venv/bin/python`
- For notebooks, use the same `.venv` as the kernel
- New terminals in VS Code should use the selected interpreter once the environment is activated

---

## Daily Workflow

Each time you come back to the project:

```bash
cd ~/Documents/GitHub/RLClass
source .venv/bin/activate
code .
```

---

## Weekly Workflow

As we add more code every week, we will add it to the `main` branch but it is IMPERATIVE you DO NOT change anything on the `main` branch, you will simply "fetch" from the main branch.

1. At the start of every week ensure you are on your branch by running `git checkout <YOUR-NAME>-branch`.
2. Fetch the latest code updates from the `main` branch with `git fetch origin`.
3. Merge the latest updates in with your code `git merge origin/main`.
4. You may now start to edit and play around with the code in your branch.
5. When you are done for the day/time being, push your edits to **your** branch.
6. On Linux/Ubuntu in VS Code, open the `Source Control` tab on the left.
7. Click the `+` button next to `Changes` to stage your changes.
8. Enter a message describing the changes you made.
9. Press **Commit** and then **Sync**.

---

## Source notes

This Ubuntu README was adapted from the uploaded class MacOS `venv` README and aligned to standard Python `venv` and VS Code interpreter selection guidance.
