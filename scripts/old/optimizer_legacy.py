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

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .sequence_io import Sequence, Alignment, SeqType
from .blast_runner import BlastRunner, BlastHit, _run_single_blast
from .alignment import (
    CenterStarAligner, 
    compute_msa_score, 
    compute_percent_identity,
    compute_column_score
)


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
    2. Optimize one parameter at a time (univariate)
    3. Fit logistic curve to find optimal value
    4. Process parameters from least to most sensitive:
       word_size -> gap_open -> gap_extend
    """
    
    # Parameter sensitivity order (least to most sensitive)
    PARAM_ORDER = ['word_size', 'gap_open', 'gap_extend']
    
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
                                   threads: int = 1,
                                   verbose: bool = False) -> float:
        """Run alignments from reference to all other sequences and return total score."""
        pairs = [
            (
                self.reference_seq.id, self.reference_seq.seq,
                other.id, other.seq,
                runner.blast_cmd, gap_open, gap_extend, word_size, evalue,
                runner.temp_dir
            )
            for other in self.other_seqs
        ]

        total = len(pairs)
        completed = 0
        hits = {}
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(_run_single_blast, pair): pair for pair in pairs}
            for future in as_completed(futures):
                completed += 1
                if verbose:
                    print(f"\r    Completed {completed}/{total} pairs", end='', flush=True)
                (id1, id2), hit_forward, hit_reverse = future.result()
                if hit_forward:
                    hits[(id1, id2)] = hit_forward
                if hit_reverse:
                    hits[(id2, id1)] = hit_reverse

        if verbose:
            print()  # newline after progress
        return self._compute_reference_score(hits)
    
    def optimize(self,
                 gap_open_range: Tuple[int, int, int],  # (start, stop, step)
                 gap_extend_range: Tuple[int, int, int],
                 word_size_range: Tuple[int, int, int],
                 evalue: float = 1e-5,
                 metric: str = 'sp_score',
                 verbose: bool = False,
                 threads: int = None) -> OptimizationResult:
        """
        Run efficient univariate parameter optimization.
        
        Args:
            gap_open_range: (start, stop, step) for gap open penalty sweep
            gap_extend_range: (start, stop, step) for gap extend penalty sweep
            word_size_range: (start, stop, step) for word size sweep
            evalue: E-value threshold
            metric: Scoring metric (used for final MSA evaluation)
            verbose: Print progress
            threads: Number of parallel BLAST workers per parameter evaluation
        
        Returns:
            OptimizationResult with best alignment and parameters
        """
        scoring_fn = get_scoring_function(metric)

        if threads is None:
            threads = max(1, multiprocessing.cpu_count() - 1)

        # Generate parameter value lists from ranges
        gap_open_values = list(range(gap_open_range[0], gap_open_range[1] + 1, gap_open_range[2]))
        gap_extend_values = list(range(gap_extend_range[0], gap_extend_range[1] + 1, gap_extend_range[2]))
        word_size_values = list(range(word_size_range[0], word_size_range[1] + 1, word_size_range[2]))
        
        # Starting defaults (middle of ranges)
        current_params = {
            'gap_open': gap_open_values[len(gap_open_values) // 2],
            'gap_extend': gap_extend_values[len(gap_extend_values) // 2],
            'word_size': word_size_values[len(word_size_values) // 2]
        }
        
        param_ranges = {
            'word_size': word_size_values,
            'gap_open': gap_open_values,
            'gap_extend': gap_extend_values
        }
        
        optimization_trace = {}
        all_results = []
        
        n_seqs = len(self.sequences)
        n_pairs = len(self.other_seqs)  # N-1 pairs
        
        if verbose:
            print(f"Univariate parameter optimization")
            print(f"  Reference sequence: {self.reference_seq.id}")
            print(f"  Aligning to {n_pairs} other sequences")
            print(f"  Parameter ranges:")
            print(f"    word_size: {word_size_values}")
            print(f"    gap_open:  {gap_open_values}")
            print(f"    gap_extend: {gap_extend_values}")
            print()
        
        with BlastRunner(self.seq_type) as runner:
            # Optimize parameters in order: least sensitive to most sensitive
            for param_name in self.PARAM_ORDER:
                param_values = param_ranges[param_name]
                
                if verbose:
                    print(f"Optimizing {param_name}...")
                
                scores = []
                
                for val in param_values:
                    # Set current parameter value
                    test_params = current_params.copy()
                    test_params[param_name] = val
                    
                    if verbose:
                        print(f"  {param_name}={val}")

                    # Run reference alignments with these parameters
                    score = self._run_reference_alignments(
                        runner,
                        gap_open=test_params['gap_open'],
                        gap_extend=test_params['gap_extend'],
                        word_size=test_params['word_size'],
                        evalue=evalue,
                        threads=threads,
                        verbose=verbose
                    )

                    scores.append(score)
                    all_results.append((test_params.copy(), score))

                    if verbose:
                        print(f"    score: {score:.1f}")
                
                # Fit logistic curve and find optimal value
                optimal_val, smoothed_score, fit_details = fit_logistic_and_find_optimum(
                    param_values, scores
                )
                
                # Update current params with optimal value
                current_params[param_name] = optimal_val
                optimization_trace[param_name] = fit_details
                
                if verbose:
                    print(f"  Optimal {param_name}: {optimal_val} "
                          f"(method: {fit_details['method']}, "
                          f"smoothed score: {smoothed_score:.1f})")
                    print()
        
        # Print final optimal parameters
        if verbose:
            print("=" * 50)
            print("OPTIMAL PARAMETERS:")
            print(f"  gap_open:   {current_params['gap_open']}")
            print(f"  gap_extend: {current_params['gap_extend']}")
            print(f"  word_size:  {current_params['word_size']}")
            print("=" * 50)
            print()
        
        # Build final MSA with optimal parameters
        if verbose:
            print("Building final MSA with optimal parameters...")
        
        with BlastRunner(self.seq_type) as runner:
            hits = runner.run_all_pairwise(
                self.sequences,
                gap_open=current_params['gap_open'],
                gap_extend=current_params['gap_extend'],
                word_size=current_params['word_size'],
                evalue=evalue,
                verbose=verbose,
                threads=threads
            )
        
        aligner = CenterStarAligner(self.sequences, self.seq_type)
        alignment = aligner.build_msa(hits, verbose=verbose)
        alignment.parameters = current_params.copy()
        
        # Score final alignment
        final_score = scoring_fn(alignment)
        alignment.score = final_score
        
        if verbose:
            print(f"Final {metric}: {final_score:.2f}")
        
        return OptimizationResult(
            best_alignment=alignment,
            best_params=current_params,
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
