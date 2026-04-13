# Stochastic Market Modeling using Markov Chains

![Status: Work in Progress](https://img.shields.io/badge/Status-Ongoing-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

## 📌 Project Overview
This project models financial market trends by constructing a Discrete-Time Markov Chain (DTMC) to simulate state transitions between Bull, Bear, and Stagnant markets. Furthermore, it implements a PageRank-indexed Retrieval-Augmented Generation (RAG) prototype that applies Markovian random walk principles to rank and extract authoritative sections from massive financial documents (SEC 10-K filings).

## 🏗️ Architectural Components

### 1. Mathematical Market Engine (DTMC)
* **Transition Matrices:** Built using historical market data to establish the probability of moving from one market state to another.
* **Steady-State Distributions:** Computing the long-term equilibrium of the market states using eigenvector calculations ($\pi = \pi P$).
* **First-Passage Times:** Calculating the expected time (number of steps) to transition into a specific target state (e.g., from Stagnant to Bull).

### 2. Graph-Based RAG Prototype (Latent Risk Retrieval)
To solve the temporal mismatch between real-time market volatility and annual reporting, this system utilizes the DTMC as a real-time trigger mechanism to query lagging fundamental data.
* **Targeted Extraction:** The system isolates "Item 1A: Risk Factors" from massive SEC 10-K filings.
* **Node Definition:** Distinct risk disclosures are treated as nodes within a directed graph.
* **PageRank Retrieval:** When the DTMC triggers a specific market state probability (e.g., an impending Bear market), the RAG simulates a Markovian random walk across the risk graph. It retrieves the most heavily interconnected and structurally critical latent vulnerabilities a company possesses, providing instant, fundamental context to statistical market shifts.

## 💻 Tech Stack
* **Core:** Python
* **Mathematics & Stats:** NumPy, SciPy (Linear Algebra)
* **Graph Algorithms:** NetworkX
* **NLP & Data:** Text Embeddings, SEC 10-K Filings Dataset

## 🚀 Current Status
This project is currently under active development. 
- [x] Architectural design and mathematical formulation
- [ ] Core DTMC engine implementation
- [ ] PageRank RAG graph construction
- [ ] Integration and testing