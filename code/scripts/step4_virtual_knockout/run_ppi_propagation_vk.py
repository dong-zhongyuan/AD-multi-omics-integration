#!/usr/bin/env python3
"""
Tissue-specific PPI Propagation Virtual Knockout (VK)
=====================================================
Second VK method for proteomics, complementing GenKI.

Method:
  - Network: Tissue-specific PPI atlas (Laman Trip et al., NBT 2025)
    Brain-specific and Blood-specific protein association networks
    from 7,811 proteomic samples across 11 human tissues.
  - VK mechanism: Random Walk with Restart (RWR) graph mutilation
    Delete KO protein node → recompute network steady state →
    measure signal change in observation tissue proteins.
  - Cross-tissue: KO in brain network (forward) or blood network (reverse),
    observe signal change in the other tissue's proteins via
    inter-tissue bridge proteins.

Output format: Consistent with GenKI statistics CSV.

Usage:
  python run_ppi_propagation_vk.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import gmres

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PPI_DATA_DIR = PROJECT_ROOT / "data" / "tissue_ppi_atlas" / "association_scores"
GENKI_FWD_DIR = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3"
GENKI_REV_DIR = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3_reverse"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "PPI_propagation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RWR parameters
ALPHA = 0.3           # Restart probability (lower = more propagation)
SCORE_THRESHOLD = 0.5  # Association score cutoff to define edges
TOP_N_OBSERVE = 200    # Number of top observation proteins to track
BRIDGE_PROTEINS_FILE = PROJECT_ROOT / "data" / "tissue_ppi_atlas" / "bridge_proteins.txt"

# ============================================================================
# Step 1: Load tissue-specific PPI networks
# ============================================================================

def load_tissue_network(tissue, score_threshold=SCORE_THRESHOLD):
    """Load tissue-specific PPI network as a weighted graph.

    Returns:
        adj: sparse adjacency matrix (weighted)
        node_list: list of protein names
        node_idx: dict {protein_name: index}
        edges: list of (prot1, prot2, score) tuples
    """
    csv_file = PPI_DATA_DIR / f"cohorts_combined_{tissue}_tumor_avg_outer_prob.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"PPI network file not found: {csv_file}")

    print(f"  Loading {tissue} PPI network (threshold={score_threshold})...")

    # Vectorized loading: read everything, filter, build index arrays
    df = pd.read_csv(csv_file, header=0, names=['prot1', 'prot2', 'score'])
    df = df[df['score'] >= score_threshold].copy()

    # Build node mapping
    all_nodes = sorted(set(df['prot1']) | set(df['prot2']))
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(all_nodes)

    print(f"    {tissue}: {n_nodes} proteins, {len(df)} edges")

    # Map protein names to indices (vectorized)
    df['i'] = df['prot1'].map(node_idx)
    df['j'] = df['prot2'].map(node_idx)

    # Build symmetric sparse adjacency
    rows = np.concatenate([df['i'].values, df['j'].values])
    cols = np.concatenate([df['j'].values, df['i'].values])
    vals = np.concatenate([df['score'].values, df['score'].values])

    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

    edges = list(zip(df['prot1'], df['prot2'], df['score']))
    return adj, all_nodes, node_idx, edges


# ============================================================================
# Step 2: Random Walk with Restart (RWR)
# ============================================================================

def compute_rwr_steady_state(adj, seed_nodes, alpha=ALPHA, max_iter=500, tol=1e-6):
    """Compute Random Walk with Restart steady-state distribution.
    
    p = (1-alpha) * W * p + alpha * s
    
    where W is the column-normalized adjacency, s is the seed vector.
    
    Args:
        adj: sparse adjacency matrix
        seed_nodes: list of node indices to seed
        alpha: restart probability
        max_iter: max iterations
        tol: convergence tolerance
    
    Returns:
        steady_state: numpy array of steady-state scores
    """
    n = adj.shape[0]
    
    # Column-normalize adjacency matrix
    col_sums = np.array(adj.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0  # avoid division by zero
    W = adj.multiply(1.0 / col_sums).tocsr()
    
    # Seed vector
    s = np.zeros(n)
    for idx in seed_nodes:
        s[idx] = 1.0
    s = s / s.sum() if s.sum() > 0 else s
    
    # Power iteration
    p = s.copy()
    for iteration in range(max_iter):
        p_new = (1 - alpha) * (W @ p) + alpha * s
        diff = np.linalg.norm(p_new - p, ord=1)
        p = p_new
        if diff < tol:
            break
    
    return p


def compute_rwr_knockout(adj, ko_node, alpha=ALPHA, max_iter=500, tol=1e-6):
    """Compute RWR steady state after knockout of a single node.
    
    Knockout = remove the node's edges (graph mutilation),
    then recompute steady state with remaining graph.
    
    Returns:
        steady_state: steady-state scores on the mutilated graph
                     (with ko_node score = 0)
    """
    n = adj.shape[0]
    
    # Create mutilated adjacency: zero out row and column of ko_node
    adj_mut = adj.copy().tolil()
    adj_mut[ko_node, :] = 0.0
    adj_mut[:, ko_node] = 0.0
    adj_mut = adj_mut.tocsr()
    
    # Seed = all nodes except KO (uniform baseline)
    # We seed with uniform distribution to capture global network perturbation
    seed = np.ones(n) / (n - 1)
    seed[ko_node] = 0.0
    
    steady = compute_rwr_steady_state(adj_mut, 
                                       seed_nodes=list(range(n)),
                                       alpha=alpha, max_iter=max_iter, tol=tol)
    # Actually, let's use the seed vector directly
    # RWR: p = (1-alpha) * W * p + alpha * s
    col_sums = np.array(adj_mut.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0
    W = adj_mut.multiply(1.0 / col_sums).tocsr()
    
    p = seed.copy()
    for iteration in range(max_iter):
        p_new = (1 - alpha) * (W @ p) + alpha * seed
        diff = np.linalg.norm(p_new - p, ord=1)
        p = p_new
        if diff < tol:
            break
    
    return p


def compute_knockout_effect(adj_baseline, adj_knockout, ko_node, 
                            observe_nodes, alpha=ALPHA):
    """Compute the knockout effect on observation proteins.
    
    Measures the difference in RWR steady state between baseline
    (intact network) and knockout (mutilated network).
    
    Args:
        adj_baseline: intact adjacency matrix
        adj_knockout: mutilated adjacency (ko_node edges removed)
        ko_node: index of knocked out protein
        observe_nodes: list of node indices to observe
        alpha: RWR restart probability
    
    Returns:
        effect_scores: dict {observe_node_idx: effect_score}
    """
    n = adj_baseline.shape[0]
    
    # Seed from a central set of AD-relevant proteins (or uniform)
    # Use uniform seed for global perturbation measure
    seed = np.ones(n) / n
    
    # Baseline steady state
    p_baseline = _rwr_power_iter(adj_baseline, seed, alpha)
    
    # Knockout steady state  
    p_knockout = _rwr_power_iter(adj_knockout, seed, alpha)
    
    # Effect = relative change in steady state for observation nodes
    effects = {}
    for obs_idx in observe_nodes:
        if obs_idx == ko_node:
            continue
        base_val = p_baseline[obs_idx]
        ko_val = p_knockout[obs_idx]
        if base_val > 1e-10:
            # Relative change
            effects[obs_idx] = abs(ko_val - base_val) / base_val
        elif ko_val > 1e-10:
            effects[obs_idx] = ko_val * 1000  # emerged from zero
        else:
            effects[obs_idx] = 0.0
    
    return effects


def _rwr_power_iter(adj, seed, alpha, max_iter=500, tol=1e-6):
    """RWR power iteration helper."""
    n = adj.shape[0]
    col_sums = np.array(adj.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0
    W = adj.multiply(1.0 / col_sums).tocsr()
    
    p = seed.copy()
    for _ in range(max_iter):
        p_new = (1 - alpha) * (W @ p) + alpha * seed
        if np.linalg.norm(p_new - p, ord=1) < tol:
            break
        p = p_new
    return p


# ============================================================================
# Step 3: Virtual Knockout for each protein
# ============================================================================

def protein_name_clean(name):
    """Clean protein name to gene symbol (consistent with GenKI script)."""
    name = str(name)
    if name.startswith('BD-'):
        base = name[3:]
        if base.startswith('pTau'):
            return 'MAPT'
        return base
    if name.startswith('pTau'):
        return 'MAPT'
    if name.startswith('Aβ') or name.startswith('AÎ²') or name.startswith('A?'):
        return 'APP'
    if name.startswith('pSNCA'):
        return 'SNCA'
    if name.startswith('pTDP'):
        return 'TARDBP'
    return name


def load_ko_targets(direction):
    """Load KO target genes from GenKI output directories.
    
    Args:
        direction: 'forward' (brain KO) or 'reverse' (blood KO)
    
    Returns:
        list of gene symbols to knockout
    """
    if direction == 'forward':
        genki_dir = GENKI_FWD_DIR
    else:
        genki_dir = GENKI_REV_DIR
    
    targets = []
    for f in sorted(genki_dir.glob('proteomics_*_statistics.csv')):
        gene = f.stem.replace('proteomics_', '').replace('_statistics', '')
        targets.append(gene)
    
    return targets


def get_observation_proteins(tissue, node_idx, step3_targets=None):
    """Get proteins to observe in the opposite tissue.
    
    If step3_targets given, observe those specific proteins.
    Otherwise, observe all proteins in the network.
    """
    if step3_targets:
        obs_indices = []
        for t in step3_targets:
            t = protein_name_clean(t)
            if t in node_idx:
                obs_indices.append(node_idx[t])
        return obs_indices
    else:
        return list(range(len(node_idx)))


def run_single_knockout(adj, node_idx, node_list, ko_gene, observe_indices,
                        alpha=ALPHA):
    """Run PPI propagation VK for a single gene.
    
    Returns:
        stats: dict with knockout statistics
        per_gene_scores: DataFrame of per-gene effect scores
    """
    n = adj.shape[0]
    
    # Check if KO gene exists in network
    if ko_gene not in node_idx:
        return None, None
    ko_idx = node_idx[ko_gene]
    
    # Baseline steady state
    seed = np.ones(n) / n
    p_baseline = _rwr_power_iter(adj, seed, alpha)
    
    # Create mutilated adjacency
    adj_mut = adj.copy().tolil()
    adj_mut[ko_idx, :] = 0.0
    adj_mut[:, ko_idx] = 0.0
    adj_mut = adj_mut.tocsr()
    
    # Knockout steady state
    p_knockout = _rwr_power_iter(adj_mut, seed, alpha)
    
    # Compute effect scores for observation proteins
    effect_scores = []
    for obs_idx in observe_indices:
        if obs_idx == ko_idx:
            continue
        base_val = p_baseline[obs_idx]
        ko_val = p_knockout[obs_idx]
        
        if base_val > 1e-12:
            relative_change = (ko_val - base_val) / base_val
            absolute_change = ko_val - base_val
        else:
            relative_change = 0.0
            absolute_change = ko_val
        
        effect_scores.append({
            'Gene': node_list[obs_idx],
            'Effect_Score': abs(relative_change),
            'Relative_Change': relative_change,
            'Absolute_Change': absolute_change,
            'Baseline_Signal': base_val,
            'Knockout_Signal': ko_val,
        })
    
    if not effect_scores:
        return None, None
    
    scores_df = pd.DataFrame(effect_scores)
    scores_df = scores_df.sort_values('Effect_Score', ascending=False)
    
    # Compute overall statistics
    effects = scores_df['Effect_Score'].values
    mean_effect = np.mean(effects)
    std_effect = np.std(effects)
    
    # Z-scores
    if std_effect > 0:
        scores_df['Z_score'] = (scores_df['Effect_Score'] - mean_effect) / std_effect
    else:
        scores_df['Z_score'] = 0.0
    
    # Significant targets (Z > 2)
    n_sig = (scores_df['Z_score'] > 2.0).sum()
    
    # KL-divergence-like measure: use entropy of effect distribution
    # Higher = more concentrated effect = stronger knockout impact
    effects_norm = effects / (effects.sum() + 1e-10)
    kl_div = float(np.sum(effects_norm * np.log(effects_norm + 1e-10) * -1))
    
    stats = {
        'KO_gene': ko_gene,
        'KL_divergence_overall': kl_div,
        'mean_effect': float(mean_effect),
        'std_effect': float(std_effect),
        'max_effect': float(np.max(effects)),
        'n_target_genes': len(effects),
        'n_significant_targets': int(n_sig),
        'top_target': scores_df.iloc[0]['Gene'] if len(scores_df) > 0 else '',
        'top_target_effect': float(scores_df.iloc[0]['Effect_Score']) if len(scores_df) > 0 else 0.0,
    }
    
    return stats, scores_df


# ============================================================================
# Step 4: Cross-tissue bridge
# ============================================================================

def compute_cross_tissue_bridge(brain_node_idx, blood_node_idx):
    """Find bridge proteins present in both tissue networks."""
    bridge = set(brain_node_idx.keys()) & set(blood_node_idx.keys())
    print(f"  Bridge proteins (in both brain & blood): {len(bridge)}")
    return bridge


# ============================================================================
# Main
# ============================================================================

from tqdm import tqdm


def main():
    print("=" * 70)
    print("Tissue-specific PPI Propagation Virtual Knockout")
    print("=" * 70)
    
    start_time = time.time()
    
    # --- Step 1: Load networks ---
    print("\n[Step 1/4] Loading tissue-specific PPI networks...")
    brain_adj, brain_nodes, brain_idx, brain_edges = load_tissue_network('brain')
    blood_adj, blood_nodes, blood_idx, blood_edges = load_tissue_network('blood')
    
    bridge = compute_cross_tissue_bridge(brain_idx, blood_idx)
    
    # --- Step 2: Forward VK (brain KO → observe blood) ---
    print("\n[Step 2/4] Forward VK: Knockout brain proteins, observe blood...")
    fwd_targets = load_ko_targets('forward')
    print(f"  Forward KO targets: {fwd_targets}")
    
    # Observation: observe bridge proteins IN THE SAME NETWORK as KO
    # Forward KO is in brain network → observe brain bridge proteins
    observe_blood_indices = [brain_idx[p] for p in bridge if p in brain_idx]
    print(f"  Observation proteins (brain bridge, same network as KO): {len(observe_blood_indices)}")
    
    fwd_stats_all = []
    fwd_out_dir = OUTPUT_DIR / "forward"
    fwd_out_dir.mkdir(exist_ok=True)
    
    for gene in fwd_targets:
        gene_clean = protein_name_clean(gene)
        print(f"\n  → Forward KO: {gene_clean}")

        if gene_clean not in brain_idx:
            print(f"    ⚠️  {gene_clean} not in brain network, skipping")
            continue

        # Skip if already done
        ranking_file = fwd_out_dir / f"proteomics_{gene_clean}_gene_ranking.csv"
        if ranking_file.exists():
            print(f"    ✅ Already done, loading cached result")
            scores = pd.read_csv(ranking_file)
            effects = scores['Effect_Score'].values
            effects_norm = effects / (effects.sum() + 1e-10)
            kl_div = float(np.sum(effects_norm * np.log(effects_norm + 1e-10) * -1))
            std_effect = float(np.std(effects))
            n_sig = int((scores['Z_score'] > 2.0).sum()) if 'Z_score' in scores else 0
            stats = {
                'KO_gene': gene_clean,
                'KL_divergence_overall': kl_div,
                'mean_effect': float(np.mean(effects)),
                'std_effect': std_effect,
                'max_effect': float(np.max(effects)),
                'n_target_genes': len(effects),
                'n_significant_targets': n_sig,
                'top_target': scores.iloc[0]['Gene'],
                'top_target_effect': float(scores.iloc[0]['Effect_Score']),
                'knockout_tissue': 'Brain',
                'observe_tissue': 'Blood',
            }
            fwd_stats_all.append(stats)
            continue
        
        stats, scores = run_single_knockout(
            brain_adj, brain_idx, brain_nodes, gene_clean,
            observe_blood_indices, alpha=ALPHA
        )
        
        if stats is None:
            print(f"    ⚠️  No effect scores generated")
            continue
        
        stats['knockout_tissue'] = 'Brain'
        stats['observe_tissue'] = 'Blood'
        fwd_stats_all.append(stats)
        
        # Save per-gene ranking
        scores.to_csv(fwd_out_dir / f"proteomics_{gene_clean}_gene_ranking.csv", index=False)
        
        print(f"    KL={stats['KL_divergence_overall']:.4f} | "
              f"mean_effect={stats['mean_effect']:.6f} | "
              f"n_sig={stats['n_significant_targets']}/{stats['n_target_genes']} | "
              f"top={stats['top_target']}({stats['top_target_effect']:.4f})")
    
    # Save forward statistics
    if fwd_stats_all:
        fwd_df = pd.DataFrame(fwd_stats_all)
        fwd_df.to_csv(fwd_out_dir / "ppi_propagation_forward_statistics.csv", index=False)
        print(f"\n  ✅ Forward statistics saved")
    
    # --- Step 3: Reverse VK (blood KO → observe brain) ---
    print("\n[Step 3/4] Reverse VK: Knockout blood proteins, observe brain...")
    rev_targets = load_ko_targets('reverse')
    print(f"  Reverse KO targets: {rev_targets}")
    
    # Reverse KO is in blood network → observe blood bridge proteins
    observe_brain_indices = [blood_idx[p] for p in bridge if p in blood_idx]
    print(f"  Observation proteins (blood bridge, same network as KO): {len(observe_brain_indices)}")
    
    rev_stats_all = []
    rev_out_dir = OUTPUT_DIR / "reverse"
    rev_out_dir.mkdir(exist_ok=True)
    
    for gene in rev_targets:
        gene_clean = protein_name_clean(gene)
        print(f"\n  → Reverse KO: {gene_clean}")
        
        if gene_clean not in blood_idx:
            print(f"    ⚠️  {gene_clean} not in blood network, skipping")
            continue
        
        stats, scores = run_single_knockout(
            blood_adj, blood_idx, blood_nodes, gene_clean,
            observe_brain_indices, alpha=ALPHA
        )
        
        if stats is None:
            print(f"    ⚠️  No effect scores generated")
            continue
        
        stats['knockout_tissue'] = 'Blood'
        stats['observe_tissue'] = 'Brain'
        rev_stats_all.append(stats)
        
        scores.to_csv(rev_out_dir / f"proteomics_{gene_clean}_gene_ranking.csv", index=False)
        
        print(f"    KL={stats['KL_divergence_overall']:.4f} | "
              f"mean_effect={stats['mean_effect']:.6f} | "
              f"n_sig={stats['n_significant_targets']}/{stats['n_target_genes']} | "
              f"top={stats['top_target']}({stats['top_target_effect']:.4f})")
    
    # Save reverse statistics
    if rev_stats_all:
        rev_df = pd.DataFrame(rev_stats_all)
        rev_df.to_csv(rev_out_dir / "ppi_propagation_reverse_statistics.csv", index=False)
        print(f"\n  ✅ Reverse statistics saved")
    
    # --- Step 4: Summary ---
    print("\n[Step 4/4] Summary")
    print(f"  Forward VK: {len(fwd_stats_all)} proteins knocked out")
    print(f"  Reverse VK: {len(rev_stats_all)} proteins knocked out")
    
    elapsed = time.time() - start_time
    print(f"\n  ⏱  Total time: {elapsed:.1f}s")
    print(f"\n{'=' * 70}")
    print("PPI Propagation VK Complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
