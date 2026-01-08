import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_peers(target_country: str, scores_df: pd.DataFrame, n_peers: int = 4) -> pd.DataFrame:
    """
    Finds the nearest peers for a target country based on Economic and Industry pillars.
    Prioritizes countries in the same region if possible (logic simplified here).
    
    Args:
        target_country: ISO code of the country to find peers for.
        scores_df: DataFrame containing 'country_code', 'economic_pillar', 'industry_pillar'.
        n_peers: Number of peers to return.
        
    Returns:
        DataFrame of peer countries with their scores and distance.
    """
    if target_country not in scores_df['country_code'].values:
        return pd.DataFrame()
    
    # diverse set of peers? or closest?
    # User asked for "closest peers"
    
    target_row = scores_df[scores_df['country_code'] == target_country].iloc[0]
    target_vector = np.array([
        target_row['economic_pillar'], 
        target_row['industry_pillar']
    ]).reshape(1, -1)
    
    # Prepare candidate pool (exclude target)
    candidates = scores_df[scores_df['country_code'] != target_country].copy()
    candidate_vectors = candidates[['economic_pillar', 'industry_pillar']].values
    
    # Use NearestNeighbors (Euclidean distance on the 2 pillars)
    nn = NearestNeighbors(n_neighbors=n_peers, metric='euclidean')
    nn.fit(candidate_vectors)
    
    distances, indices = nn.kneighbors(target_vector)
    
    peers = candidates.iloc[indices[0]].copy()
    peers['distance'] = distances[0]
    
    return peers
