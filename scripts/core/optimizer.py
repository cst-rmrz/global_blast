"""
Parameter optimization module for BLAST MSA

Performs grid search over BLAST parameters to find optimal alignment.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from itertools import product
import sys

from .sequence_io import Sequence, Alignment, SeqType
from .blast_runner import BlastRunner, BlastHit
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


class ParameterOptimizer:
    """
    Optimizes BLAST parameters through grid search.
    
    Tests combinations of gap_open, gap_extend, and word_size
    parameters to find the combination producing the highest-scoring MSA.
    """
    
    def __init__(self, sequences: List[Sequence], seq_type: SeqType):
        self.sequences = sequences
        self.seq_type = seq_type
    
    def optimize(self,
                 gap_open_values: List[int],
                 gap_extend_values: List[int],
                 word_size_values: Optional[List[int]] = None,
                 evalue: float = 1e-5,
                 metric: str = 'sp_score',
                 verbose: bool = False,
                 threads: int = None) -> OptimizationResult:
        """
        Run parameter optimization.
        
        Args:
            gap_open_values: List of gap open penalties to test
            gap_extend_values: List of gap extend penalties to test  
            word_size_values: List of word sizes to test (None = use default)
            evalue: E-value threshold
            metric: Scoring metric ('sp_score', 'percent_identity', 'column_score')
            verbose: Print progress
        
        Returns:
            OptimizationResult with best alignment and parameters
        """
        scoring_fn = get_scoring_function(metric)
        
        # Default word sizes if not specified
        if word_size_values is None:
            if self.seq_type == SeqType.PROTEIN:
                word_size_values = [3]
            else:
                word_size_values = [11]
        
        # Generate parameter combinations
        param_combinations = list(product(
            gap_open_values,
            gap_extend_values,
            word_size_values
        ))
        
        n_combinations = len(param_combinations)
        
        if verbose:
            print(f"Testing {n_combinations} parameter combinations...")
            print(f"  Gap open: {gap_open_values}")
            print(f"  Gap extend: {gap_extend_values}")
            print(f"  Word size: {word_size_values}")
            print(f"  Metric: {metric}")
            print()
        
        best_alignment = None
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        with BlastRunner(self.seq_type) as runner:
            for i, (gap_open, gap_extend, word_size) in enumerate(param_combinations):
                params = {
                    'gap_open': gap_open,
                    'gap_extend': gap_extend,
                    'word_size': word_size
                }
                
                if verbose:
                    print(f"\rTesting {i+1}/{n_combinations}: "
                          f"gap_open={gap_open}, gap_extend={gap_extend}, "
                          f"word_size={word_size}", end='')
                    sys.stdout.flush()
                
                # Run all pairwise BLASTs with these parameters
                hits = runner.run_all_pairwise(
                    self.sequences,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    word_size=word_size,
                    evalue=evalue,
                    verbose=False,
                    threads=threads
                )
                
                # Build MSA
                aligner = CenterStarAligner(self.sequences, self.seq_type)
                alignment = aligner.build_msa(hits, verbose=False)
                
                # Score alignment
                score = scoring_fn(alignment)
                all_results.append((params.copy(), score))
                
                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_alignment = alignment
                    best_alignment.parameters = params.copy()
                    best_alignment.score = score
                
                if verbose:
                    print(f" -> score: {score:.2f}")
        
        if verbose:
            print()
            print(f"Best parameters: gap_open={best_params['gap_open']}, "
                  f"gap_extend={best_params['gap_extend']}, "
                  f"word_size={best_params['word_size']}")
            print(f"Best {metric}: {best_score:.2f}")
        
        return OptimizationResult(
            best_alignment=best_alignment,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            metric_used=metric
        )


def parse_param_list(value: str) -> List[int]:
    """Parse comma-separated parameter list from config file"""
    if not value or value.strip() == '':
        return []
    return [int(x.strip()) for x in value.split(',') if x.strip()]
