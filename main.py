import yfinance as yf
from src.dtmc_engine import MarketDTMC
from src.risk_graph_rag import RiskGraphRAG

# ==========================================
# 1. Dummy SEC 10-K Data for the Prototype
# ==========================================
# In production, this is extracted from the 2.8GB SEC dataset in your /data folder.
DUMMY_10K_RISK_FACTORS = """
We rely heavily on a small number of suppliers in Southeast Asia for our semiconductor components. 
A disruption in this supply chain could severely impact our manufacturing capabilities. 
Furthermore, our high levels of corporate debt require us to maintain strict liquidity ratios. 
If global interest rates rise significantly, our debt servicing costs will increase, reducing net income. 
The semiconductor supply chain is also vulnerable to geopolitical tensions which could halt exports overnight. 
Failure to maintain our liquidity ratios could trigger a default on our primary credit facility.
"""

def main():
    print("==================================================")
    print(" STOCHASTIC MARKET MODELING & LATENT RISK RAG")
    print("==================================================\n")

    # 1. Initialize Engines
    dtmc = MarketDTMC(threshold=0.005) # 0.5% daily volatility threshold
    rag = RiskGraphRAG(similarity_threshold=0.3)

    # 2. Fetch Real-World Market Data
    print("[Market] Fetching S&P 500 data for transition matrix calibration...")
    sp500 = yf.download('^GSPC', period='2y', interval='1d', progress=False)
    prices = sp500['Close'].squeeze() # Get 1D array of closing prices

    # 3. Fit the Markov Chain
    transition_matrix, steady_state = dtmc.fit(prices)
    latest_return = prices.pct_change().iloc[-1]
    current_state = dtmc.get_current_state(latest_return)

    print(f"\n--- DTMC Market Analysis ---")
    print(f"Current Market State: {current_state.upper()} (Latest Return: {latest_return*100:.2f}%)")
    print(f"Long-Term Steady State Equilibrium: {steady_state}\n")

    # 4. Trigger Logic: If market is Bearish, pull latent risks
    # (For demonstration, we will force the trigger to show how the RAG works)
    print("--- Simulating Bear Market Trigger ---")
    print("Triggering PageRank Graph Search on SEC 10-K Item 1A...\n")
    
    rag.build_graph(DUMMY_10K_RISK_FACTORS)
    top_risks = rag.run_pagerank_retrieval(top_k=2)

    print("\n[RAG] Top Retrieved Latent Risks (Ranked by PageRank Centrality):")
    for i, (risk, score) in enumerate(top_risks, 1):
        print(f"{i}. (Score: {score}) {risk}")

if __name__ == "__main__":
    main()