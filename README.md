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
   ```bash
   ./install_sc2.sh
   ```

2. **Build the Docker image**  
   Navigate to the `/docker` folder and run:
   ```bash
   docker build --network=host --no-cache -t pymarl:1.0 .
   ```

3. **Run the Docker container**
   ```bash
   ./run.sh 0
   ```

4. **Start training (inside the Docker container)**
   ```bash
   python3 src/main.py --config=hygma --env-config=sc2 with env_args.map_name=3s_vs_5z
   ```

---

## Notes

- Ensure you have **NVIDIA GPU drivers** and **NVIDIA Container Toolkit** installed for GPU acceleration in Docker.
- The `install_sc2.sh` script will download the StarCraft II Linux package and SMAC maps.
- Large game files (StarCraft II and maps) are excluded from version control via `.gitignore`.

