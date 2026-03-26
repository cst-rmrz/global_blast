"""
Parameter optimization module for BLAST MSA

Implements efficient univariate parameter optimization with logistic curve fitting.
Instead of brute-force grid search, this:
1. Uses only the first sequence as reference (N-1 alignments instead of N*(N-1)/2)
2. Sweeps parameters one at a time (univariate)
3. Fits a logistic curve to find the optimal inflection point
4. Processes parameters from least to most sensitive
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from itertools import product
import sys
import numpy as np
from scipy.optimize import curve_fit

from .sequence_io import Sequence, Alignment, SeqType
from .blast_runner import BlastRunner, BlastHit
from .alignment import (
    CenterStarAligner, 
    compute_msa_score, 
    compute_percent_identity,
    compute_column_score
)


# Valid gap penalty combinations for BLOSUM62 (blastp)
# From BLAST documentation
VALID_BLOSUM62_GAPS = {
    (11, 2), (10, 2), (9, 2), (8, 2), (7, 2), (6, 2),
    (13, 1), (12, 1), (11, 1), (10, 1), (9, 1)
}


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_alignment: Alignment
    best_params: Dict[str, int]
    best_score: float
    all_results: List[Tuple[Dict[str, int], float]]
    metric_used: str
    optimization_trace: Optional[Dict] = None  # Store per-parameter optimization details


def get_scoring_function(metric: str) -> Callable[[Alignment], float]:
    """Get the scoring function for a given metric name"""
    metrics = {
        'sp_score': compute_msa_score,
        'percent_identity': compute_percent_identity,
        'column_score': compute_column_score,
    }
    
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}. "
                        f"Available: {', '.join(metrics.keys())}")
    
    return metrics[metric]


def is_valid_gap_penalty(gap_open: int, gap_extend: int, seq_type: SeqType) -> bool:
    """
    Check if a gap penalty combination is valid for the sequence type.
    
    Protein (BLOSUM62) has strict constraints.
    Nucleotide is more flexible but has some constraints too.
    """
    if seq_type == SeqType.PROTEIN:
        return (gap_open, gap_extend) in VALID_BLOSUM62_GAPS
    else:
        # Nucleotide: valid combinations depend on scoring (1,-2 default)
        # Generally: gap_open + gap_extend * 2 should work
        # Safe combinations for blastn with default scoring
        valid_nucl = {
            (5, 2), (4, 2), (2, 2), (1, 2), (0, 2),
            (4, 4), (2, 4), (0, 4),
            (3, 1), (2, 1), (1, 1)
        }
        return (gap_open, gap_extend) in valid_nucl


def get_valid_gap_combinations(seq_type: SeqType) -> List[Tuple[int, int]]:
    """Get all valid (gap_open, gap_extend) combinations for a sequence type."""
    if seq_type == SeqType.PROTEIN:
        return sorted(list(VALID_BLOSUM62_GAPS), key=lambda x: (x[1], x[0]))
    else:
        return [
            (5, 2), (4, 2), (2, 2), (1, 2), (0, 2),
            (3, 1), (2, 1), (1, 1)
        ]


def logistic_function(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """
    Logistic function for curve fitting.
    
    L: maximum value (asymptote)
    k: steepness of curve
    x0: x-value of sigmoid midpoint
    b: minimum value (baseline)
    """
    return b + L / (1 + np.exp(-k * (x - x0)))


def fit_logistic_and_find_optimum(param_values: List[int], 
                                   scores: List[float]) -> Tuple[int, float, Dict]:
    """
    Fit a logistic curve to parameter-score data and find optimal parameter.
    
    For small sample sizes (<10), falls back to selecting max score directly.
    In case of ties, returns the average parameter value.
    
    Returns:
        (optimal_param, smoothed_max_score, fit_details)
    """
    x = np.array(param_values, dtype=float)
    y = np.array(scores, dtype=float)
    
    fit_details = {
        'method': None,
        'param_values': param_values,
        'scores': scores,
        'logistic_params': None,
        'r_squared': None
    }
    
    # For small samples or if all scores are identical, use direct max
    if len(x) < 6 or np.std(y) < 1e-10:
        max_score = np.max(y)
        max_indices = np.where(y == max_score)[0]
        
        if len(max_indices) > 1:
            # Tie: take average of tied parameter values
            optimal_param = int(np.round(np.mean(x[max_indices])))
            fit_details['method'] = 'max_with_tie_average'
        else:
            optimal_param = int(x[max_indices[0]])
            fit_details['method'] = 'direct_max'
        
        return optimal_param, max_score, fit_details
    
    # Try to fit logistic curve
    try:
        # Initial guesses
        L_init = np.max(y) - np.min(y)
        b_init = np.min(y)
        x0_init = np.mean(x)
        k_init = 1.0
        
        # Bounds to keep parameters reasonable
        bounds = (
            [0, 0.001, np.min(x), -np.inf],  # lower bounds
            [np.inf, 10, np.max(x), np.inf]   # upper bounds
        )
        
        popt, pcov = curve_fit(
            logistic_function, x, y,
            p0=[L_init, k_init, x0_init, b_init],
            bounds=bounds,
            maxfev=5000
        )
        
        L, k, x0, b = popt
        
        # Calculate R-squared
        y_pred = logistic_function(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        fit_details['logistic_params'] = {'L': L, 'k': k, 'x0': x0, 'b': b}
        fit_details['r_squared'] = r_squared
        
        # If fit is poor (R² < 0.5), fall back to direct max
        if r_squared < 0.5:
            fit_details['method'] = 'direct_max_poor_fit'
            max_score = np.max(y)
            max_indices = np.where(y == max_score)[0]
            if len(max_indices) > 1:
                optimal_param = int(np.round(np.mean(x[max_indices])))
            else:
                optimal_param = int(x[max_indices[0]])
            return optimal_param, max_score, fit_details
        
        # Find optimal parameter from fitted curve
        # Use the asymptotic maximum (L + b) as the smoothed max score
        smoothed_max = L + b
        
        # Find parameter value that achieves ~95% of max (on the plateau)
        # This is more robust than using x0 directly
        x_fine = np.linspace(np.min(x), np.max(x), 100)
        y_fine = logistic_function(x_fine, *popt)
        
        # Find where we reach 95% of the range
        threshold = b + 0.95 * L
        plateau_indices = np.where(y_fine >= threshold)[0]
        
        if len(plateau_indices) > 0:
            # Take the first parameter value that reaches the plateau
            optimal_x = x_fine[plateau_indices[0]]
        else:
            # Fallback: use the parameter with highest predicted score
            optimal_x = x_fine[np.argmax(y_fine)]
        
        # Round to nearest actual parameter value
        optimal_param = int(round(optimal_x))
        # Clamp to tested range
        optimal_param = max(min(param_values), min(max(param_values), optimal_param))
        
        fit_details['method'] = 'logistic_fit'
        
        return optimal_param, smoothed_max, fit_details
        
    except (RuntimeError, ValueError) as e:
        # Curve fitting failed, fall back to direct max
        fit_details['method'] = 'direct_max_fit_failed'
        fit_details['error'] = str(e)
        
        max_score = np.max(y)
        max_indices = np.where(y == max_score)[0]
        if len(max_indices) > 1:
            optimal_param = int(np.round(np.mean(x[max_indices])))
        else:
            optimal_param = int(x[max_indices[0]])
        
        return optimal_param, max_score, fit_details


class ParameterOptimizer:
    """
    Optimizes BLAST parameters using efficient univariate sweeps.
    
    Strategy:
    1. Use first sequence as reference, align only to others (N-1 pairs)
    2. Optimize parameters jointly where they have constraints (gap penalties)
    3. Fit logistic curve to find optimal value
    4. Process: word_size first, then gap penalties together
    """
    
    def __init__(self, sequences: List[Sequence], seq_type: SeqType):
        self.sequences = sequences
        self.seq_type = seq_type
        self.reference_seq = sequences[0]  # First sequence is reference
        self.other_seqs = sequences[1:]
    
    def _compute_reference_score(self, hits: Dict[Tuple[str, str], BlastHit]) -> float:
        """
        Compute total alignment score using only reference-to-others alignments.
        
        Uses sum of bitscores from reference sequence to all others.
        """
        total_score = 0.0
        ref_id = self.reference_seq.id
        
        for other in self.other_seqs:
            # Check both directions
            hit1 = hits.get((ref_id, other.id))
            hit2 = hits.get((other.id, ref_id))
            
            # Take best hit from either direction
            if hit1 and hit2:
                total_score += max(hit1.bitscore, hit2.bitscore)
            elif hit1:
                total_score += hit1.bitscore
            elif hit2:
                total_score += hit2.bitscore
            # If no hit, contributes 0
        
        return total_score
    
    def _run_reference_alignments(self, runner: BlastRunner,
                                   gap_open: int, gap_extend: int,
                                   word_size: int, evalue: float,
                                   threads: int = None) -> float:
        """
        Run alignments from reference to all other sequences using database approach.
        
        Creates DB from reference, BLASTs all others against it in one call.
        Much faster than individual pairwise BLASTs.
        """
        hits = runner.run_reference_vs_others(
            self.reference_seq, self.other_seqs,
            gap_open, gap_extend, word_size, evalue, threads
        )
        return self._compute_reference_score(hits)
    
    def optimize(self,
                 gap_open_range: Tuple[int, int, int],  # (start, stop, step)
                 gap_extend_range: Tuple[int, int, int],
                 word_size_range: Tuple[int, int, int],
                 evalue: float = 1e-5,
                 metric: str = 'sp_score',
                 verbose: bool = False,
                 threads: int = None,
                 aligner_class=None) -> OptimizationResult:
        """
        Run efficient parameter optimization.
        
        For protein sequences, gap penalties have strict constraints (BLOSUM62),
        so we test valid combinations rather than independent sweeps.
        
        For nucleotide sequences, we also use valid combinations.
        """
        scoring_fn = get_scoring_function(metric)
        
        # Generate word size values from range
        word_size_values = list(range(word_size_range[0], word_size_range[1] + 1, word_size_range[2]))
        
        # Get valid gap penalty combinations for this sequence type
        valid_gap_combos = get_valid_gap_combinations(self.seq_type)
        
        # Filter to requested ranges
        gap_open_min, gap_open_max = gap_open_range[0], gap_open_range[1]
        gap_extend_min, gap_extend_max = gap_extend_range[0], gap_extend_range[1]
        
        filtered_gap_combos = [
            (go, ge) for go, ge in valid_gap_combos
            if gap_open_min <= go <= gap_open_max and gap_extend_min <= ge <= gap_extend_max
        ]
        
        if not filtered_gap_combos:
            # Fall back to the full valid set if filter is too restrictive
            filtered_gap_combos = valid_gap_combos
            if verbose:
                print(f"  Warning: Requested gap ranges have no valid combinations.")
                print(f"  Using all valid combinations for {self.seq_type.name}.")
        
        optimization_trace = {}
        all_results = []
        
        n_pairs = len(self.other_seqs)  # N-1 pairs
        
        if verbose:
            print(f"Univariate parameter optimization")
            print(f"  Reference sequence: {self.reference_seq.id}")
            print(f"  Aligning to {n_pairs} other sequences")
            print(f"  Parameter values:")
            print(f"    word_size: {word_size_values}")
            print(f"    gap penalties (open, extend): {filtered_gap_combos}")
            print()
        
        # Start with middle values
        current_word_size = word_size_values[len(word_size_values) // 2]
        current_gap_open, current_gap_extend = filtered_gap_combos[len(filtered_gap_combos) // 2]
        
        with BlastRunner(self.seq_type) as runner:
            # 1. Optimize word_size first (least sensitive)
            if verbose:
                print(f"Optimizing word_size...")
            
            word_size_scores = []
            for ws in word_size_values:
                if verbose:
                    print(f"  word_size={ws}", end='', flush=True)
                
                score = self._run_reference_alignments(
                    runner,
                    gap_open=current_gap_open,
                    gap_extend=current_gap_extend,
                    word_size=ws,
                    evalue=evalue,
                    threads=threads
                )
                word_size_scores.append(score)
                all_results.append(({'word_size': ws, 'gap_open': current_gap_open, 
                                    'gap_extend': current_gap_extend}, score))
                
                if verbose:
                    print(f" -> score: {score:.1f}")
            
            optimal_ws, _, ws_fit = fit_logistic_and_find_optimum(word_size_values, word_size_scores)
            current_word_size = optimal_ws
            optimization_trace['word_size'] = ws_fit
            
            if verbose:
                print(f"  Optimal word_size: {optimal_ws} (method: {ws_fit['method']})")
                print()
            
            # 2. Optimize gap penalties together (they're interdependent)
            if verbose:
                print(f"Optimizing gap penalties...")
            
            gap_scores = []
            for gap_open, gap_extend in filtered_gap_combos:
                if verbose:
                    print(f"  gap_open={gap_open}, gap_extend={gap_extend}", end='', flush=True)
                
                score = self._run_reference_alignments(
                    runner,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    word_size=current_word_size,
                    evalue=evalue,
                    threads=threads
                )
                gap_scores.append(score)
                all_results.append(({'word_size': current_word_size, 'gap_open': gap_open,
                                    'gap_extend': gap_extend}, score))
                
                if verbose:
                    print(f" -> score: {score:.1f}")
            
            # Find best gap penalty combination
            best_idx = np.argmax(gap_scores)
            best_score = gap_scores[best_idx]
            
            # Check for ties
            max_indices = [i for i, s in enumerate(gap_scores) if s == best_score]
            if len(max_indices) > 1:
                # Average the tied combinations
                avg_gap_open = int(np.round(np.mean([filtered_gap_combos[i][0] for i in max_indices])))
                avg_gap_extend = int(np.round(np.mean([filtered_gap_combos[i][1] for i in max_indices])))
                # Find closest valid combination
                best_combo = min(filtered_gap_combos, 
                                key=lambda x: abs(x[0] - avg_gap_open) + abs(x[1] - avg_gap_extend))
                method = 'max_with_tie_average'
            else:
                best_combo = filtered_gap_combos[best_idx]
                method = 'direct_max'
            
            current_gap_open, current_gap_extend = best_combo
            optimization_trace['gap_penalties'] = {
                'method': method,
                'combinations': filtered_gap_combos,
                'scores': gap_scores,
                'best_combo': best_combo
            }
            
            if verbose:
                print(f"  Optimal gap_open={current_gap_open}, gap_extend={current_gap_extend} "
                      f"(method: {method})")
                print()
        
        # Final optimal parameters
        best_params = {
            'gap_open': current_gap_open,
            'gap_extend': current_gap_extend,
            'word_size': current_word_size
        }
        
        # Print final optimal parameters
        if verbose:
            print("=" * 50)
            print("OPTIMAL PARAMETERS:")
            print(f"  gap_open:   {best_params['gap_open']}")
            print(f"  gap_extend: {best_params['gap_extend']}")
            print(f"  word_size:  {best_params['word_size']}")
            print("=" * 50)
            print()
        
        # Build final MSA with optimal parameters
        if verbose:
            print("Building final MSA with optimal parameters...")
        
        with BlastRunner(self.seq_type) as runner:
            hits = runner.run_all_pairwise(
                self.sequences,
                gap_open=best_params['gap_open'],
                gap_extend=best_params['gap_extend'],
                word_size=best_params['word_size'],
                evalue=evalue,
                verbose=verbose,
                threads=threads
            )
        
        if aligner_class is None:
            aligner_class = CenterStarAligner
        aligner = aligner_class(self.sequences, self.seq_type)
        alignment = aligner.build_msa(hits, verbose=verbose)
        alignment.parameters = best_params.copy()
        
        # Score final alignment
        final_score = scoring_fn(alignment)
        alignment.score = final_score
        
        if verbose:
            print(f"Final {metric}: {final_score:.2f}")
        
        return OptimizationResult(
            best_alignment=alignment,
            best_params=best_params,
            best_score=final_score,
            all_results=all_results,
            metric_used=metric,
            optimization_trace=optimization_trace
        )


def parse_param_list(value: str) -> List[int]:
    """Parse comma-separated parameter list from config file"""
    if not value or value.strip() == '':
        return []
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_param_range(value: str) -> Tuple[int, int, int]:
    """
    Parse parameter range from config file.
    
    Format: "start:stop:step" e.g., "2:10:1"
    Returns: (start, stop, step)
    """
    parts = value.strip().split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid range format '{value}'. Expected 'start:stop:step'")
    return (int(parts[0]), int(parts[1]), int(parts[2]))
