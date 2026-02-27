#!/usr/bin/env python3
"""
BLAST MSA - Multiple Sequence Alignment using BLAST

A tool that uses the BLAST algorithm for sequence alignment instead of
traditional global aligners like ClustalW or MUSCLE.

Usage:
    blast_msa.py input.fasta -o output.phy
    blast_msa.py input.fasta -o output.fasta --optimize
    blast_msa.py input.fasta -o output.aln -f clustal --params my_params.params
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import configparser
import io

# Add script directory to path for imports (handles running from any location)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from scripts import (
    parse_fasta,
    detect_sequence_type,
    write_alignment,
    format_from_extension,
    SeqType,
    BlastRunner,
    CenterStarAligner,
    ParameterOptimizer,
    parse_param_list,
    parse_param_range,
    compute_msa_score,
    compute_percent_identity,
)


def load_params(params_file: Path) -> Dict[str, Any]:
    """
    Load parameters from a .params file.
    
    Uses INI-style parsing but handles the headerless format.
    """
    params = {
        # BLAST parameters (sequence-type specific)
        'gap_open_protein': 11,
        'gap_open_nucleotide': 5,
        'gap_extend_protein': 1,
        'gap_extend_nucleotide': 2,
        'word_size_protein': 3,
        'word_size_nucleotide': 11,
        'evalue': 1e-5,
        # Optimization parameters (sequence-type specific) - now ranges
        'optimize_gap_open_protein': (5, 15, 2),
        'optimize_gap_open_nucleotide': (2, 6, 1),
        'optimize_gap_extend_protein': (1, 3, 1),
        'optimize_gap_extend_nucleotide': (1, 2, 1),
        'optimize_word_size_protein': (2, 3, 1),
        'optimize_word_size_nucleotide': (7, 15, 2),
        'optimize_metric': 'sp_score',
        # Extension parameters
        'extend_termini': True,
        'terminal_gap_penalty': 0,
        # Output parameters
        'default_format': 'fasta',
        'wrap_width': 80,
    }
    
    if not params_file.exists():
        return params
    
    # Read file and add a dummy section header for configparser
    content = params_file.read_text()
    
    # Parse manually (simpler than configparser for this format)
    for line in content.split('\n'):
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#') or line.startswith('['):
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Parse based on expected type
            if key in ['gap_open_protein', 'gap_open_nucleotide',
                       'gap_extend_protein', 'gap_extend_nucleotide',
                       'word_size_protein', 'word_size_nucleotide', 
                       'terminal_gap_penalty', 'wrap_width']:
                params[key] = int(value)
            elif key == 'evalue':
                params[key] = float(value)
            elif key == 'extend_termini':
                params[key] = value.lower() in ('true', 'yes', '1')
            elif key.startswith('optimize_') and key != 'optimize_metric':
                # Parse range format "start:stop:step"
                params[key] = parse_param_range(value)
            else:
                params[key] = value
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description='Multiple Sequence Alignment using BLAST algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sequences.fasta -o aligned.phy
  %(prog)s sequences.fasta -o aligned.fasta --optimize
  %(prog)s proteins.faa -o aligned.aln -f clustal
  %(prog)s sequences.fasta -o aligned.nex -f nexus --optimize --metric percent_identity

Output formats:
  fasta          FASTA format (default)
  phylip         Phylip strict format (10-char names)
  phylip-relaxed Phylip relaxed format (longer names)
  clustal        Clustal format with conservation marks
  nexus          NEXUS format for phylogenetics software
        """
    )
    
    parser.add_argument('input', type=Path,
                       help='Input FASTA file')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output alignment file')
    parser.add_argument('-f', '--format', type=str, default=None,
                       choices=['fasta', 'phylip', 'phylip-relaxed', 'clustal', 'nexus'],
                       help='Output format (default: infer from extension)')
    parser.add_argument('--params', type=Path, default=None,
                       help='Parameters file (default: blast.params in script directory)')
    
    # BLAST parameters (override params file)
    parser.add_argument('--gap-open', type=int, default=None,
                       help='Gap opening penalty')
    parser.add_argument('--gap-extend', type=int, default=None,
                       help='Gap extension penalty')
    parser.add_argument('--word-size', type=int, default=None,
                       help='Word size for BLAST seeding')
    parser.add_argument('--evalue', type=float, default=None,
                       help='E-value threshold')
    
    # Optimization
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize parameters through grid search')
    parser.add_argument('--metric', type=str, default=None,
                       choices=['sp_score', 'percent_identity', 'column_score'],
                       help='Scoring metric for optimization (default: sp_score)')
    
    # Output options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--stats', action='store_true',
                       help='Print alignment statistics')
    parser.add_argument('-t', '--threads', type=int, default=None,
                       help='Number of parallel threads (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Load parameters
    if args.params:
        params_file = args.params
    else:
        params_file = SCRIPT_DIR / 'blast.params'
    
    params = load_params(params_file)
    
    if args.verbose:
        print(f"BLAST MSA - Multiple Sequence Alignment using BLAST")
        print(f"=" * 50)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print()
    
    # Parse input sequences
    if args.verbose:
        print("Reading sequences...")
    
    sequences = parse_fasta(args.input)
    
    if len(sequences) < 2:
        print(f"Error: Need at least 2 sequences for alignment, found {len(sequences)}", 
              file=sys.stderr)
        sys.exit(1)
    
    # Detect sequence type
    seq_type = detect_sequence_type(sequences)
    
    if args.verbose:
        print(f"  Found {len(sequences)} sequences")
        print(f"  Sequence type: {'protein' if seq_type == SeqType.PROTEIN else 'nucleotide'}")
        print()
    
    # Set up BLAST parameters (sequence-type specific)
    is_protein = seq_type == SeqType.PROTEIN
    
    word_size = args.word_size
    if word_size is None:
        word_size = (params['word_size_protein'] if is_protein 
                    else params['word_size_nucleotide'])
    
    gap_open = args.gap_open
    if gap_open is None:
        gap_open = (params['gap_open_protein'] if is_protein 
                   else params['gap_open_nucleotide'])
    
    gap_extend = args.gap_extend
    if gap_extend is None:
        gap_extend = (params['gap_extend_protein'] if is_protein 
                     else params['gap_extend_nucleotide'])
    
    evalue = args.evalue if args.evalue is not None else params['evalue']
    
    # Run alignment
    if args.optimize:
        if args.verbose:
            print("Running parameter optimization...")
            print()
        
        # Get optimization parameters (sequence-type specific) - now ranges
        if is_protein:
            opt_gap_open = params['optimize_gap_open_protein']
            opt_gap_extend = params['optimize_gap_extend_protein']
            opt_word_size = params['optimize_word_size_protein']
        else:
            opt_gap_open = params['optimize_gap_open_nucleotide']
            opt_gap_extend = params['optimize_gap_extend_nucleotide']
            opt_word_size = params['optimize_word_size_nucleotide']
        
        metric = args.metric if args.metric else params['optimize_metric']
        
        optimizer = ParameterOptimizer(sequences, seq_type)
        result = optimizer.optimize(
            gap_open_range=opt_gap_open,
            gap_extend_range=opt_gap_extend,
            word_size_range=opt_word_size,
            evalue=evalue,
            metric=metric,
            verbose=args.verbose,
            threads=args.threads
        )
        
        alignment = result.best_alignment
        
        if args.verbose:
            print()
            print("Optimization complete!")
            print(f"  Best parameters: {result.best_params}")
            print(f"  Best {metric}: {result.best_score:.2f}")
            print()
    
    else:
        if args.verbose:
            print("Running BLAST alignments...")
            print(f"  Gap open: {gap_open}")
            print(f"  Gap extend: {gap_extend}")
            print(f"  Word size: {word_size}")
            print()
        
        with BlastRunner(seq_type) as runner:
            hits = runner.run_all_pairwise(
                sequences,
                gap_open=gap_open,
                gap_extend=gap_extend,
                word_size=word_size,
                evalue=evalue,
                verbose=args.verbose,
                threads=args.threads
            )
        
        if args.verbose:
            print(f"  Found {len(hits)} pairwise alignments")
            print()
            print("Building MSA...")
        
        aligner = CenterStarAligner(sequences, seq_type)
        alignment = aligner.build_msa(hits, verbose=args.verbose)
        alignment.parameters = {
            'gap_open': gap_open,
            'gap_extend': gap_extend,
            'word_size': word_size
        }
    
    # Validate alignment
    if not alignment.is_valid():
        print("Error: Generated alignment is invalid (length mismatch)", file=sys.stderr)
        sys.exit(1)
    
    # Determine output format
    output_format = args.format
    if output_format is None:
        output_format = format_from_extension(args.output)
    
    # Write output
    if args.verbose:
        print(f"Writing alignment to {args.output} ({output_format} format)...")
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    write_alignment(alignment, args.output, format=output_format, 
                   wrap_width=params['wrap_width'])
    
    # Print statistics
    if args.stats or args.verbose:
        print()
        print("Alignment statistics:")
        print(f"  Sequences: {alignment.n_seqs}")
        print(f"  Alignment length: {alignment.length}")
        print(f"  Sum-of-pairs score: {compute_msa_score(alignment):.2f}")
        print(f"  Average percent identity: {compute_percent_identity(alignment):.1f}%")
    
    if args.verbose:
        print()
        print("Done!")


if __name__ == '__main__':
    main()
