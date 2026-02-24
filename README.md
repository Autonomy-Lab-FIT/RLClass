# RLClass
Code for Dr Tiwari's Reinforcement Learning Class
## Install 
1. Ensure you have Python 3.11 installed on your system. (We have only verified the code works for Python 3.11, but 3.10+ should work fine.) [https://www.python.org/downloads/](https://www.python.org/downloads/) **If you are installing Python for the first time, ennsure you select 'Add to PATH' before finishing the installation!**
2. Install Conda or Miniconda for your operating system to create your virtual environment. [https://www.anaconda.com/docs/getting-started/miniconda/main ](https://www.anaconda.com/docs/getting-started/miniconda/install) You have to make an account, use your school account. **Ensure you add Conda or MiniConda to your environment PATH during installation!**
3. (Optional) We personally prefer using VS Code for our coding editor, but if you are more familiar with something like PyCharm or Sublime you can use that. In order to install VS Code follow the instructions for your operating system here: [https://code.visualstudio.com/ ](https://code.visualstudio.com/)
4. Launch your code editor. (from here on out, we will assume you are using VS Code.)
5. When you launch VS Code, go to Extensions on the left taskbar and install the following extensions: <br>
   a. Python <br>
   b. Python Debugger <br>
   c. Pylance <br>
   d. Jupyter <br>
6. You will now need to install Git [https://git-scm.com/install/](https://git-scm.com/install/) for your system. If using Linux or Mac you can now move onto the step 8. If you are using Windows you will need to install GitHub Desktop [https://desktop.github.com/download/](https://desktop.github.com/download/) in order to pull from Github.
7. Next you will clone the repo from GitHub. Go to [https://github.com/Autonomy-Lab-FIT/RLClass](https://github.com/Autonomy-Lab-FIT/RLClass) and click the green dropdown button that says "<> Code". There will be a link to click that says "Open with GitHub Desktop", click that and confirm the location of the repo on your system. (It is easier to just leave it in the `Documents\Github` directory.
8. Once you have Git set up and the repo cloned, go into VS Code and press `Ctrl + K + O` and select the RLClass folder to open the Workspace.
9. Open a Command Prompt terminal in VS Code and create your own branch with `git checkout -b <YOUR-NAME>-branch`.
10. Push your newly created branch to the GutHub repo with `git push -u origin <YOUR-NAME>-branch`

## Installing Requirements

Once you have created your branch and pushed it to the repo, you are now ready to install the requirements and begin coding. 
1. Open a terminal in the VS Code editor and make sure your directory is pointing to the `RLClass/` folder.
2. Create your conda virtual environment with `conda create --name <YOUR-ENV-NAME> python=3.11` **The python version is very important, without it your environment will not have Python!**
3. Once it is created, activate your environment with `conda activate <YOUR-ENV-NAME>`
4. Verify the Python version with `python --version`. It should say 3.11.xx
5. Install the requirements by runnning 'pip install -r requirements.txt`
6. Once this is complete you can open the first Jupyter Notebook. You will need to select a Kernel from the top right. Click the button that says "Select Kernel".
7. On the menu that pops up, select "Python Environments". You should see your conda env, select that one.
8. If you get a pop up saying that you need to install 'ipykernel' install that and follow the prompts.
9. You should now be able to run the Jupyter Notebooks and code. 

## Weekly Workflow

As we add more code every week, we will add it to the `main` branch but it is IMPERATIVE you DO NOT change anything on the `main` branch, you will simply "fetch" from the main branch. 
1. At the start of every week ensure you are on your branch by running `git checkout <YOUR-NAME>-branch`.
2. Fetch the latest code updates from the `main` branch with `git fetch origin`.
3. Merge the latest updates in with your code `git merge origin/main`.
4. You may now start to edit and play around with the code in your branch.
5. When you are done for the day/time being, it is important to push your edits to YOUR branch. For Windows proceed to step 9. On Linux/Mac click the `Source Control` button on the left-hand taskbar.
6. Click the '+' button next to 'Changes' to stage your changes. <img width="328" height="473" alt="image" src="https://github.com/user-attachments/assets/4146e689-1797-4b69-bef6-038714a1fcaa" />

7. Enter a message in the textbox describing the changes you made.
8. Press the button that says 'Commit' and then press it again when it changes to 'Sync'. You have now pushed your edits to the cloud on your branch.
9. On Windows, return back to GitHub Desktop. Enter a commit message in the textbox on the lower left-hand side that says 'Summary' and a description in the description box.
10. Press the button that says 'Commit' to commit your edits.
11. A new blue button should then appear in the middle saying to 'push origin' your changes, click that and you are done.

