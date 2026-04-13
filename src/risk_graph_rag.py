import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class RiskGraphRAG:
    def __init__(self, similarity_threshold: float = 0.5):
        """
        PageRank-indexed Graph RAG for retrieving latent corporate risks.
        :param similarity_threshold: Minimum cosine similarity to create an edge between text nodes.
        """
        self.threshold = similarity_threshold
        # Using a fast, local embedding model (no API keys needed)
        print("[System] Loading local NLP embedding model (this takes a moment the first time)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph = nx.Graph()
        self.chunks = []

    def build_graph(self, text_corpus: str):
        """Chunks the 10-K text, embeds it, and constructs the Markovian similarity graph."""
        # Simple chunking: split by periods (in production, use a recursive text splitter)
        self.chunks = [chunk.strip() for chunk in text_corpus.split('.') if len(chunk.strip()) > 30]
        
        if not self.chunks:
            raise ValueError("Text corpus is empty or chunks are too small.")

        # 1. Generate text embeddings
        print(f"[System] Embedding {len(self.chunks)} risk chunks...")
        embeddings = self.embedder.encode(self.chunks)

        # 2. Calculate pairwise Cosine Similarity
        sim_matrix = cosine_similarity(embeddings)

        # 3. Build the Graph
        for i in range(len(self.chunks)):
            self.graph.add_node(i, text=self.chunks[i])
            for j in range(i + 1, len(self.chunks)):
                if sim_matrix[i, j] >= self.threshold:
                    # Create an edge weighted by their semantic similarity
                    self.graph.add_edge(i, j, weight=sim_matrix[i, j])
                    
        print(f"[System] Risk Graph Built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges.")

    def run_pagerank_retrieval(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Simulates a random walk on the text graph to find the most central, authoritative risk factors.
        """
        if len(self.graph.nodes) == 0:
            return []

        # Run PageRank (which is literally a Markov Chain steady-state calculation on the graph)
        pagerank_scores = nx.pagerank(self.graph, weight='weight')

        # Sort nodes by their PageRank score in descending order
        ranked_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        # Retrieve the text for the top_k nodes
        top_risks = []
        for node_id, score in ranked_nodes[:top_k]:
            top_risks.append((self.chunks[node_id], round(score, 4)))

        return top_risks