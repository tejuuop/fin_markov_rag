import numpy as np
import pandas as pd
from typing import Tuple, Dict

class MarketDTMC:
    def __init__(self, threshold: float = 0.005):
        """
        Discrete-Time Markov Chain for Market States.
        :param threshold: Percentage return threshold to classify Bull/Bear states (e.g., 0.005 = 0.5%)
        """
        self.threshold = threshold
        self.states = ['Bear', 'Stagnant', 'Bull']
        self.transition_matrix = None
        self.steady_state = None

    def _classify_returns(self, returns: pd.Series) -> pd.Series:
        """Classifies numerical returns into discrete Markov states."""
        conditions = [
            (returns < -self.threshold),
            (returns > self.threshold)
        ]
        choices = ['Bear', 'Bull']
        # Default to Stagnant if it doesn't break the threshold
        return pd.Series(np.select(conditions, choices, default='Stagnant'))

    def fit(self, prices: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculates the Transition Matrix and the Steady-State Distribution.
        """
        # 1. Calculate daily percentage returns
        returns = prices.pct_change().dropna()
        state_sequence = self._classify_returns(returns)

        # 2. Build Transition Matrix (P)
        matrix = pd.crosstab(state_sequence.shift(), state_sequence, normalize='index')
        
        # Ensure all states are present in the matrix to prevent dimensionality errors
        for state in self.states:
            if state not in matrix.columns:
                matrix[state] = 0.0
            if state not in matrix.index:
                matrix.loc[state] = 0.0
        
        # Reorder to match self.states exactly
        matrix = matrix.reindex(index=self.states, columns=self.states).fillna(0.0)
        self.transition_matrix = matrix.values

        # 3. Calculate Steady-State Distribution (pi * P = pi)
        # We solve this by finding the eigenvector of P transpose corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find the index of the eigenvalue closest to 1.0
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum() # Normalize so probabilities sum to 1

        self.steady_state = {state: round(prob, 4) for state, prob in zip(self.states, pi)}
        
        return self.transition_matrix, self.steady_state

    def get_current_state(self, latest_return: float) -> str:
        """Returns the immediate market state based on the latest close."""
        if latest_return < -self.threshold: return 'Bear'
        if latest_return > self.threshold: return 'Bull'
        return 'Stagnant'