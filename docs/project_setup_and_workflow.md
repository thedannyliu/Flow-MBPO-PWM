# Project Setup and Development Workflow

This document outlines the Git structure and development workflow for the Flow-MBPO-PWM project.

## 1. Project Structure

This project uses a nested Git repository structure managed via **Git Submodules**.

- **`Flow-MBPO-PWM/` (Superproject)**: This is the main repository for the research project. It acts as a "meta-repository" or an experiment logbook. Its primary roles are:
  - To hold documentation (like this file).
  - To store experiment results and analysis scripts.
  - To "pin" the specific version of the `PWM` source code used for any given experiment.

- **`PWM/` (Submodule)**: This is a Git submodule that contains the actual implementation of the PWM algorithm.
  - It is a fork of the original [PWM repository](https://github.com/nicklashansen/pwm).
  - All source code modifications, feature additions (like Flow-Matching dynamics), and bug fixes happen exclusively within this repository.

This structure cleanly separates the core codebase from the high-level experiment management.

## 2. Core Development Workflow

The workflow involves two loops: an "inner loop" for code changes and an "outer loop" for snapshotting the project state.

### Part 1: Code Development (The Inner Loop)

All code changes happen inside the `PWM` submodule.

**Scenario**: You want to add a new feature, like the Flow-Matching model.

1.  **Navigate into the submodule:**
    ```bash
    cd PWM
    ```

2.  **Create a new branch for your feature:** It's good practice to keep `main` clean and work on feature branches.
    ```bash
    # Make sure you are on the main branch and up-to-date first
    git checkout main
    git pull origin main

    # Create your new feature branch
    git checkout -b dev/flow-dynamics
    ```

3.  **Write your code:** Add, edit, and delete files as needed.
    ```bash
    # e.g., create the new dynamics model file
    touch src/pwm/models/flow_dynamics.py
    # ...add content to the file...
    ```

4.  **Commit your changes within the submodule:**
    ```bash
    git add .
    git commit -m "feat: Add initial scaffold for FlowDynamics model"
    ```

5.  **Push your branch to your fork (`origin`):**
    ```bash
    git push origin dev/flow-dynamics
    ```

At this point, your code changes are safely stored in the `PWM` repository's history. The superproject is not yet aware of these changes.

### Part 2: Snapshotting the Experiment State (The Outer Loop)

After you've committed code changes in the submodule, you update the superproject to create a snapshot.

1.  **Navigate back to the superproject root:**
    ```bash
    cd ..
    ```

2.  **Check the status:** You will see that Git has detected new commits in the submodule.
    ```bash
    git status
    # On branch master
    # Changes not staged for commit:
    #   (use "git add <file>..." to update what will be committed)
    #
    #	modified:   PWM (new commits)
    #
    ```

3.  **Stage the submodule update:** This tells the superproject to pin the newer version of the `PWM` code.
    ```bash
    git add PWM
    ```

4.  **Commit the snapshot:** The commit message should describe the high-level state of the experiment.
    ```bash
    git commit -m "Experiment: Update PWM submodule with FlowDynamics scaffold"
    ```

5.  **Push the snapshot to the superproject remote:**
    ```bash
    git push
    ```

## 3. Syncing with the Original PWM Repository

To keep your fork updated with bug fixes or features from the original authors:

1.  **Navigate into the submodule:**
    ```bash
    cd PWM
    ```
2.  **Add the original repository as a remote named `upstream` (only needs to be done once):**
    ```bash
    git remote add upstream https://github.com/nicklashansen/pwm.git
    ```
3.  **Fetch updates from `upstream`:**
    ```bash
    git fetch upstream
    ```
4.  **Merge the updates into your main branch:**
    ```bash
    git checkout main
    git merge upstream/main
    # Resolve any conflicts if they occur
    ```
5.  **Push the updated main branch to your fork:**
    ```bash
    git push origin main
    ```

## 4. Cloning the Project in the Future

To clone this entire project, including the submodule, on a new machine, use the `--recurse-submodules` flag:

```bash
git clone --recurse-submodules git@github.com:thedannyliu/Flow-MBPO-PWM.git
```
