# Yonsei University AI Major - 2025 Project Code

This repository contains the code for the 2025 project of the AI Major at Yonsei University.  
It is based on the [HYGMA paper](https://openreview.net/pdf?id=mgJkeqc685), with additional algorithms implemented by our team.

## Paper Reference

> **HYGMA: Hypergraph Coordination Networks with Dynamic Grouping for Multi-Agent Reinforcement Learning**  
> *ICLR 2025 Conference Paper.*

Link: [https://openreview.net/pdf?id=mgJkeqc685](https://openreview.net/pdf?id=mgJkeqc685)

---

## Setup Instructions

After downloading the code, please follow these steps:

1. **Install the StarCraft II simulator**  
   Navigate to the root folder( `/marl-dalija`) and run:
   ```bash
   ./install_sc2.sh
   ```

2. **Build the Docker image**  
   Navigate to the `/docker` folder and run:
   ```bash
   docker build --network=host --no-cache -t pymarl:1.0 .
   ```

3. **Launch the Docker container**  
   Navigate to the root folder( `/marl-dalija`) and run:
   ```bash
   ./run.sh 0
   ```

4. **Enter the container's bash shell**  
   After step 3, exit the container if it doesn't automatically enter bash, and run:
    ```bash
    ./run.sh 0
    ```
   This will start the container in bash mode.

5. **Start training (inside the Docker container)**   
   ```bash
   python3 src/main.py --config=hygma --env-config=sc2 with env_args.map_name=3s_vs_5z
   ```

---

## 🔐 Cloning this private repository

If you are a collaborator and the repository is private, please use SSH:

```bash
git clone git@github.com:rohyoungsang/marl-dalija.git
```

Make sure you have added your SSH key to your GitHub account.  
See [GitHub Docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) for instructions.

If you must use HTTPS and are prompted for authentication, generate and use a **personal access token (PAT)** instead of your password:  
[Create a GitHub PAT](https://github.com/settings/tokens)

```bash
git clone https://github.com/rohyoungsang/marl-dalija.git
```
When prompted, use your GitHub **username** and the **token** as your password.

---

## Notes

- Ensure you have **NVIDIA GPU drivers** and **NVIDIA Container Toolkit** installed for GPU acceleration in Docker.
- The `install_sc2.sh` script will download the StarCraft II Linux package and SMAC maps.
- Large game files (StarCraft II and maps) are excluded from version control via `.gitignore`.

