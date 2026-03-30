"""
Graph/structural feature extraction from propagation cascades.
Total output: 65-dimensional feature vector per news cascade.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional
from collections import Counter


def build_cascade_graph(tweet_ids: List[str]) -> nx.DiGraph:
    """
    Build a simple cascade graph from tweet IDs.
    In a full implementation, edges come from retweet relationships.
    Here we create a placeholder chain structure from tweet IDs.
    """
    G = nx.DiGraph()
    for i, tid in enumerate(tweet_ids):
        G.add_node(tid, timestamp=i)
        if i > 0:
            # Placeholder: chain structure; real data uses retweet parent info
            G.add_edge(tweet_ids[i - 1], tid, weight=1.0)
    return G


def extract_cascade_features(G: nx.DiGraph) -> np.ndarray:
    """
    Extract 25 cascade-level features from a propagation graph.

    Features:
    - Depth metrics (4): max_depth, avg_depth, max_breadth, branching_factor
    - Temporal dynamics (6): velocity_early, velocity_peak, time_to_peak,
                             acceleration, decay_rate, burst_count
    - Structural virality (1): V score
    - Size/reach (4): total_users, total_posts, participation_ratio, audience_size
    - Reshare patterns (4): avg_time_to_reshare, reshare_depth_corr,
                            max_breadth_level, breadth_depth_ratio
    - Padding (6): reserved for future features
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(25, dtype=np.float32)

    # Find root (node with no predecessors)
    roots = [node for node in G.nodes() if G.in_degree(node) == 0]
    root = roots[0] if roots else list(G.nodes())[0]

    # Depth metrics
    try:
        lengths = nx.single_source_shortest_path_length(G, root)
        depths = list(lengths.values())
        max_depth = max(depths) if depths else 0
        avg_depth = np.mean(depths) if depths else 0

        # Breadth at each level
        level_counts = Counter(depths)
        max_breadth = max(level_counts.values()) if level_counts else 0
        max_breadth_level = max(level_counts, key=level_counts.get) if level_counts else 0
    except nx.NetworkXError:
        max_depth, avg_depth, max_breadth, max_breadth_level = 0, 0, 0, 0

    # Branching factor
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    branching_factor = np.mean([d for d in out_degrees if d > 0]) if any(d > 0 for d in out_degrees) else 0

    # Temporal dynamics (using node index as time proxy)
    timestamps = sorted(nx.get_node_attributes(G, "timestamp").values())
    if len(timestamps) > 1:
        velocity_early = min(len(timestamps), 10)  # First 10 nodes
        velocity_peak = max(1, n // max(max_depth, 1))
        time_to_peak = len(timestamps) // 2
        acceleration = velocity_peak / max(time_to_peak, 1)
        decay_rate = max(0, 1.0 - (velocity_peak / max(n, 1)))
        burst_count = sum(1 for i in range(1, len(timestamps)) if timestamps[i] - timestamps[i-1] == 0)
    else:
        velocity_early, velocity_peak, time_to_peak = 0, 0, 0
        acceleration, decay_rate, burst_count = 0, 0, 0

    # Structural virality (average shortest path length)
    try:
        if n > 1 and nx.is_weakly_connected(G):
            undirected = G.to_undirected()
            structural_virality = nx.average_shortest_path_length(undirected)
        else:
            structural_virality = 0
    except Exception:
        structural_virality = 0

    # Size & reach
    total_users = n
    total_posts = G.number_of_edges()
    participation_ratio = total_users / max(total_posts, 1)
    audience_size = np.log1p(n * 100)  # Placeholder; real: sum of followers

    # Reshare patterns
    avg_time_to_reshare = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
    reshare_depth_corr = np.corrcoef(range(len(timestamps)), timestamps)[0, 1] if len(timestamps) > 2 else 0
    breadth_depth_ratio = max_breadth / max(max_depth, 1)

    features = np.array([
        max_depth, avg_depth, max_breadth, branching_factor,
        velocity_early, velocity_peak, time_to_peak,
        acceleration, decay_rate, burst_count,
        structural_virality,
        total_users, total_posts, participation_ratio, audience_size,
        avg_time_to_reshare, reshare_depth_corr if not np.isnan(reshare_depth_corr) else 0,
        max_breadth_level, breadth_depth_ratio,
        0, 0, 0, 0, 0, 0,  # Reserved
    ], dtype=np.float32)

    return features


def extract_user_features(user_data: Optional[Dict] = None) -> np.ndarray:
    """
    Extract 20 user-level features.

    Features include profile characteristics, behavior patterns,
    network position metrics, and bot indicators.
    """
    if user_data is None:
        return np.zeros(20, dtype=np.float32)

    followers = np.log1p(user_data.get("followers_count", 0))
    friends = np.log1p(user_data.get("friends_count", 0))
    ff_ratio = followers / max(friends, 0.01)
    statuses = np.log1p(user_data.get("statuses_count", 0))
    account_age = np.log1p(user_data.get("account_age_days", 0))
    verified = float(user_data.get("verified", False))
    has_desc = float(bool(user_data.get("description", "")))
    desc_len = len(user_data.get("description", ""))
    default_img = float(user_data.get("default_profile_image", True))

    # Behavior (placeholders for when timeline data is available)
    posting_freq = user_data.get("posting_frequency", 0)
    retweet_ratio = user_data.get("retweet_ratio", 0.5)
    avg_rt_received = np.log1p(user_data.get("avg_retweets_received", 0))
    avg_fav_received = np.log1p(user_data.get("avg_favorites_received", 0))

    # Network position (computed from graph)
    degree_centrality = user_data.get("degree_centrality", 0)
    pagerank = user_data.get("pagerank", 0)
    betweenness = user_data.get("betweenness", 0)
    clustering = user_data.get("clustering_coefficient", 0)

    # Bot indicators
    bot_score = user_data.get("bot_score", 0)
    coordination_score = user_data.get("coordination_score", 0)
    amplification = user_data.get("amplification_factor", 0)

    return np.array([
        followers, friends, ff_ratio, statuses, account_age,
        verified, has_desc, desc_len, default_img,
        posting_freq, retweet_ratio, avg_rt_received, avg_fav_received,
        degree_centrality, pagerank, betweenness, clustering,
        bot_score, coordination_score, amplification,
    ], dtype=np.float32)


def extract_community_features(G: nx.Graph) -> np.ndarray:
    """Extract 8 community-level features using Louvain-style detection."""
    if G.number_of_nodes() < 3:
        return np.zeros(8, dtype=np.float32)

    undirected = G.to_undirected() if G.is_directed() else G

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(undirected))
        num_communities = len(communities)
        modularity = nx.algorithms.community.modularity(undirected, communities)

        sizes = [len(c) for c in communities]
        max_community = max(sizes) / G.number_of_nodes()
        size_gini = gini_coefficient(sizes)

        # Cross-community edges
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
        cross_edges = sum(
            1 for u, v in undirected.edges()
            if community_map.get(u, -1) != community_map.get(v, -2)
        )
        cross_ratio = cross_edges / max(undirected.number_of_edges(), 1)

        polarization = 1.0 - cross_ratio
    except Exception:
        num_communities, modularity = 1, 0
        max_community, size_gini = 1.0, 0
        cross_ratio, polarization = 0, 0

    return np.array([
        num_communities, modularity, max_community, size_gini,
        cross_ratio, polarization, 0, 0,  # 2 reserved
    ], dtype=np.float32)


def extract_temporal_features(timestamps: List[float]) -> np.ndarray:
    """Extract 12 temporal features from cascade timestamps."""
    if len(timestamps) < 2:
        return np.zeros(12, dtype=np.float32)

    ts = np.array(sorted(timestamps))
    ts_relative = ts - ts[0]

    # 24-hour activity vector (simplified to 4 buckets)
    hour_bins = np.zeros(4)
    for t in ts_relative:
        bucket = min(int(t / 6) % 4, 3)
        hour_bins[bucket] += 1
    hour_bins = hour_bins / max(hour_bins.sum(), 1)

    # 7-day activity (simplified to 4 buckets)
    day_bins = np.zeros(4)
    for t in ts_relative:
        bucket = min(int(t / (24 * 7 / 4)), 3)
        day_bins[bucket] += 1
    day_bins = day_bins / max(day_bins.sum(), 1)

    # Inter-event times
    inter_times = np.diff(ts_relative)
    iet_std = np.std(inter_times) if len(inter_times) > 0 else 0

    # Synchronized burst score
    burst_threshold = np.percentile(inter_times, 10) if len(inter_times) > 0 else 0
    burst_score = np.mean(inter_times < burst_threshold) if len(inter_times) > 0 else 0

    features = np.concatenate([hour_bins, day_bins, [iet_std, burst_score, 0, 0]])
    return features.astype(np.float32)


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient for a list of values."""
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))
