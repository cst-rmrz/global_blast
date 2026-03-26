"""
BLAST runner module for BLAST MSA

Handles:
- Creating temporary BLAST databases
- Running BLAST with specified parameters
- Parsing BLAST output into usable alignment data
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

from .sequence_io import Sequence, SeqType


@dataclass
class BlastHit:
    """Single BLAST alignment result"""
    query_id: str
    subject_id: str
    query_start: int  # 1-based
    query_end: int
    subject_start: int  # 1-based
    subject_end: int
    query_seq: str  # Aligned sequence with gaps
    subject_seq: str  # Aligned sequence with gaps
    evalue: float
    bitscore: float
    identity: float  # Percent identity
    
    @property
    def query_len_aligned(self) -> int:
        return self.query_end - self.query_start + 1
    
    @property
    def subject_len_aligned(self) -> int:
        return self.subject_end - self.subject_start + 1


def _run_single_blast(args: Tuple) -> Tuple[Tuple[str, str], Optional[BlastHit], Optional[BlastHit]]:
    """
    Worker function for parallel BLAST execution.
    
    Args:
        args: Tuple of (seq1_id, seq1_seq, seq2_id, seq2_seq, blast_cmd, gap_open, gap_extend, word_size, evalue, temp_dir)
    
    Returns:
        Tuple of ((id1, id2), forward_hit, reverse_hit)
    """
    (seq1_id, seq1_seq, seq2_id, seq2_seq, 
     blast_cmd, gap_open, gap_extend, word_size, evalue, temp_dir) = args
    
    # Create unique temp files for this pair
    import uuid
    pair_id = uuid.uuid4().hex[:8]
    query_path = Path(temp_dir) / f'query_{pair_id}.fasta'
    subject_path = Path(temp_dir) / f'subject_{pair_id}.fasta'
    
    try:
        # Write sequences
        with open(query_path, 'w') as f:
            f.write(f">{seq1_id}\n{seq1_seq}\n")
        with open(subject_path, 'w') as f:
            f.write(f">{seq2_id}\n{seq2_seq}\n")
        
        # Run forward BLAST (seq1 vs seq2)
        hit_forward = _execute_blast(
            query_path, subject_path, blast_cmd,
            gap_open, gap_extend, word_size, evalue
        )
        
        # Run reverse BLAST (seq2 vs seq1)
        hit_reverse = _execute_blast(
            subject_path, query_path, blast_cmd,
            gap_open, gap_extend, word_size, evalue
        )
        
        return ((seq1_id, seq2_id), hit_forward, hit_reverse)
    
    finally:
        # Clean up temp files
        if query_path.exists():
            query_path.unlink()
        if subject_path.exists():
            subject_path.unlink()


def _execute_blast(query_path: Path, subject_path: Path, blast_cmd: str,
                   gap_open: int, gap_extend: int, word_size: int,
                   evalue: float, max_hsps: int = 1) -> Optional[BlastHit]:
    """Execute a single BLAST command and parse result."""
    cmd = [
        blast_cmd,
        '-query', str(query_path),
        '-subject', str(subject_path),
        '-gapopen', str(gap_open),
        '-gapextend', str(gap_extend),
        '-word_size', str(word_size),
        '-evalue', str(evalue),
        '-outfmt', '6 qseqid sseqid qstart qend sstart send qseq sseq evalue bitscore pident qlen slen',
        '-max_target_seqs', '1',
        '-max_hsps', str(max_hsps)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return None

    if max_hsps == 1:
        line = output.split('\n')[0]
        fields = line.split('\t')

        if len(fields) < 11:
            return None

        return BlastHit(
            query_id=fields[0],
            subject_id=fields[1],
            query_start=int(fields[2]),
            query_end=int(fields[3]),
            subject_start=int(fields[4]),
            subject_end=int(fields[5]),
            query_seq=fields[6],
            subject_seq=fields[7],
            evalue=float(fields[8]),
            bitscore=float(fields[9]),
            identity=float(fields[10])
        )

    # Multi-HSP mode: pick the HSP with best query coverage
    best_hit = None
    best_coverage = 0.0

    for line in output.split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) < 11:
            continue

        hit = BlastHit(
            query_id=fields[0],
            subject_id=fields[1],
            query_start=int(fields[2]),
            query_end=int(fields[3]),
            subject_start=int(fields[4]),
            subject_end=int(fields[5]),
            query_seq=fields[6],
            subject_seq=fields[7],
            evalue=float(fields[8]),
            bitscore=float(fields[9]),
            identity=float(fields[10])
        )

        # Compute query coverage from qlen field if available, else from alignment span
        if len(fields) >= 13:
            qlen = int(fields[11])
            coverage = hit.query_len_aligned / qlen if qlen > 0 else 0.0
        else:
            coverage = hit.query_len_aligned  # fallback: prefer longer alignments

        if coverage > best_coverage:
            best_coverage = coverage
            best_hit = hit

    return best_hit


def _retry_single_pair(args: Tuple) -> Tuple[Tuple[str, str], Optional[BlastHit], Optional[BlastHit]]:
    """
    Worker function for parallel coverage-guard retries.

    Args:
        args: Tuple of (id1, seq1, id2, seq2, blast_cmd, gap_open, gap_extend,
              word_size, evalue, temp_dir)

    Returns:
        Tuple of ((id1, id2), forward_hit, reverse_hit)
    """
    (id1, seq1, id2, seq2,
     blast_cmd, gap_open, gap_extend, word_size, evalue, temp_dir) = args

    import uuid
    pair_id = uuid.uuid4().hex[:8]
    query_path = Path(temp_dir) / f'retry_{pair_id}_q.fasta'
    subject_path = Path(temp_dir) / f'retry_{pair_id}_s.fasta'

    try:
        with open(query_path, 'w') as f:
            f.write(f">{id1}\n{seq1}\n")
        with open(subject_path, 'w') as f:
            f.write(f">{id2}\n{seq2}\n")

        fwd = _execute_blast(
            query_path, subject_path, blast_cmd,
            gap_open, gap_extend, word_size, evalue, max_hsps=3
        )
        rev = _execute_blast(
            subject_path, query_path, blast_cmd,
            gap_open, gap_extend, word_size, evalue, max_hsps=3
        )

        return ((id1, id2), fwd, rev)

    finally:
        if query_path.exists():
            query_path.unlink()
        if subject_path.exists():
            subject_path.unlink()


def _parse_blast_output(output: str) -> List[BlastHit]:
    """Parse tabular BLAST output into list of BlastHit objects."""
    hits = []
    for line in output.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) < 11:
            continue
        hits.append(BlastHit(
            query_id=fields[0],
            subject_id=fields[1],
            query_start=int(fields[2]),
            query_end=int(fields[3]),
            subject_start=int(fields[4]),
            subject_end=int(fields[5]),
            query_seq=fields[6],
            subject_seq=fields[7],
            evalue=float(fields[8]),
            bitscore=float(fields[9]),
            identity=float(fields[10])
        ))
    return hits


class BlastRunner:
    """
    Manages BLAST execution for MSA construction.
    
    Creates temporary databases and runs pairwise BLASTs.
    """
    
    def __init__(self, seq_type: SeqType):
        self.seq_type = seq_type
        self.temp_dir = None
        self.db_path = None
        
        # Determine BLAST programs based on sequence type
        if seq_type == SeqType.PROTEIN:
            self.blast_cmd = 'blastp'
            self.makedb_type = 'prot'
        else:
            self.blast_cmd = 'blastn'
            self.makedb_type = 'nucl'
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='blast_msa_')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _check_blast_installed(self) -> bool:
        """Check if BLAST+ is available"""
        try:
            subprocess.run(['makeblastdb', '-version'], 
                         capture_output=True, check=True)
            subprocess.run([self.blast_cmd, '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def create_database(self, sequences: List[Sequence], db_name: str = 'blastdb') -> Path:
        """Create a BLAST database from sequences."""
        if not self._check_blast_installed():
            raise RuntimeError(
                "BLAST+ not found. Please install NCBI BLAST+ tools.\n"
                "Ubuntu/Debian: sudo apt install ncbi-blast+\n"
                "macOS: brew install blast\n"
                "Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
            )
        
        # Write sequences to temp FASTA
        fasta_path = Path(self.temp_dir) / f'{db_name}.fasta'
        with open(fasta_path, 'w') as f:
            for seq in sequences:
                f.write(f">{seq.id}\n{seq.seq}\n")
        
        # Create BLAST database
        db_path = Path(self.temp_dir) / db_name
        
        cmd = [
            'makeblastdb',
            '-in', str(fasta_path),
            '-dbtype', self.makedb_type,
            '-out', str(db_path),
            '-parse_seqids'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"makeblastdb failed: {result.stderr}")
        
        return db_path
    
    def run_against_database(self, queries: List[Sequence], db_path: Path,
                              gap_open: int, gap_extend: int, word_size: int,
                              evalue: float, threads: int = 1) -> Dict[Tuple[str, str], BlastHit]:
        """
        Run all query sequences against a BLAST database in a single call.
        
        This is much more efficient than individual pairwise BLASTs:
        - Single subprocess spawn
        - BLAST handles internal parallelization
        - Database index is built once and reused
        
        Returns:
            Dictionary mapping (query_id, subject_id) to BlastHit
        """
        if not queries:
            return {}
        
        # Write all queries to single FASTA
        query_path = Path(self.temp_dir) / 'queries.fasta'
        with open(query_path, 'w') as f:
            for seq in queries:
                f.write(f">{seq.id}\n{seq.seq}\n")
        
        # Run BLAST with all queries at once
        # Use max_target_seqs equal to number of sequences to get all hits
        n_seqs = len(queries)
        cmd = [
            self.blast_cmd,
            '-query', str(query_path),
            '-db', str(db_path),
            '-gapopen', str(gap_open),
            '-gapextend', str(gap_extend),
            '-word_size', str(word_size),
            '-evalue', str(evalue),
            '-outfmt', '6 qseqid sseqid qstart qend sstart send qseq sseq evalue bitscore pident qlen slen',
            '-max_target_seqs', str(max(n_seqs + 5, 10)),  # Get enough hits
            '-max_hsps', '1',
            '-num_threads', str(threads)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 and 'error' in result.stderr.lower():
            raise RuntimeError(f"BLAST failed: {result.stderr}")
        
        # Parse results - keep best hit per query-subject pair
        hits = {}
        for hit in _parse_blast_output(result.stdout):
            # Skip self-hits
            if hit.query_id == hit.subject_id:
                continue
            
            key = (hit.query_id, hit.subject_id)
            # Keep hit with best bitscore if we see the same pair twice
            if key not in hits or hit.bitscore > hits[key].bitscore:
                hits[key] = hit
        
        return hits
    
    def run_reference_vs_others(self, reference: Sequence, others: List[Sequence],
                                 gap_open: int, gap_extend: int, word_size: int,
                                 evalue: float, threads: int = None) -> Dict[Tuple[str, str], BlastHit]:
        """
        Efficiently align a reference sequence against all others.
        
        Creates a database from the reference, then BLASTs all others against it.
        Also runs the reverse (others as DB, reference as query) to get both directions.
        
        This is optimized for the center-star MSA approach and parameter optimization.
        """
        if threads is None:
            threads = max(1, multiprocessing.cpu_count() - 1)
        
        hits = {}
        
        # Forward: others query against reference DB
        ref_db = self.create_database([reference], db_name='ref_db')
        forward_hits = self.run_against_database(
            others, ref_db, gap_open, gap_extend, word_size, evalue, threads
        )
        hits.update(forward_hits)
        
        # Reverse: reference query against others DB
        others_db = self.create_database(others, db_name='others_db')
        reverse_hits = self.run_against_database(
            [reference], others_db, gap_open, gap_extend, word_size, evalue, threads
        )
        hits.update(reverse_hits)
        
        return hits
    
    def run_pairwise(self, query: Sequence, subject: Sequence,
                     gap_open: int = 11, gap_extend: int = 1,
                     word_size: int = None, evalue: float = 1e-5) -> Optional[BlastHit]:
        """
        Run pairwise BLAST between two sequences.
        
        Returns the best hit, or None if no significant alignment found.
        """
        # Write query to temp file
        query_path = Path(self.temp_dir) / 'query.fasta'
        with open(query_path, 'w') as f:
            f.write(f">{query.id}\n{query.seq}\n")
        
        # Write subject to temp file (as database for this pair)
        subject_path = Path(self.temp_dir) / 'subject.fasta'
        with open(subject_path, 'w') as f:
            f.write(f">{subject.id}\n{subject.seq}\n")
        
        # Set default word size if not specified
        if word_size is None:
            word_size = 3 if self.seq_type == SeqType.PROTEIN else 11
        
        return _execute_blast(
            query_path, subject_path, self.blast_cmd,
            gap_open, gap_extend, word_size, evalue
        )
    
    def run_all_pairwise(self, sequences: List[Sequence],
                         gap_open: int = 11, gap_extend: int = 1,
                         word_size: int = None, evalue: float = 1e-5,
                         verbose: bool = False,
                         threads: int = None,
                         coverage_threshold: float = 0.5) -> Dict[Tuple[str, str], BlastHit]:
        """
        Run all pairwise BLASTs between sequences.

        Uses database approach: creates DB of all sequences, BLASTs each against it.
        BLAST handles parallelization internally with -num_threads.

        After initial BLAST, checks alignment coverage for each hit. Hits covering
        less than coverage_threshold of the query are retried with more sensitive
        parameters. Set coverage_threshold=0 to disable the coverage guard.

        Returns a dictionary mapping (query_id, subject_id) tuples to BlastHit objects.
        """
        if not self._check_blast_installed():
            raise RuntimeError("BLAST+ not found.")

        # Set default word size if not specified
        if word_size is None:
            word_size = 3 if self.seq_type == SeqType.PROTEIN else 11

        # Determine number of threads
        if threads is None:
            threads = max(1, multiprocessing.cpu_count() - 1)

        n_seqs = len(sequences)
        total_pairs = n_seqs * (n_seqs - 1)  # Both directions

        if verbose:
            print(f"  Creating BLAST database...")

        # Create database of all sequences
        all_db = self.create_database(sequences, db_name='all_seqs_db')

        if verbose:
            print(f"  Running all-vs-all BLAST with {threads} threads...")

        # BLAST all sequences against the database
        hits = self.run_against_database(
            sequences, all_db, gap_open, gap_extend, word_size, evalue, threads
        )

        if verbose:
            print(f"  Found {len(hits)} pairwise alignments")

        # Coverage guard: retry low-coverage hits with more sensitive parameters
        if coverage_threshold > 0:
            seq_lens = {s.id: len(s.seq) for s in sequences}
            hits = self._retry_low_coverage_hits(
                hits, sequences, seq_lens, word_size, evalue,
                coverage_threshold, verbose
            )

        return hits

    def run_all_pairwise_multi_hsp(self, sequences: List[Sequence],
                                    gap_open: int = 11, gap_extend: int = 1,
                                    word_size: int = None, evalue: float = 1e-5,
                                    verbose: bool = False,
                                    threads: int = None,
                                    max_hsps: int = 10) -> Dict[Tuple[str, str], List[BlastHit]]:
        """
        Like run_all_pairwise() but returns ALL HSPs per (query, subject) pair.

        Multi-HSP mode is important for sequences that have large insertions
        relative to other sequences: BLAST finds multiple conserved blocks as
        separate HSPs. chain_blast_hsps() can then chain them into a complete
        alignment that correctly places the insertion.

        Returns: Dict mapping (query_id, subject_id) -> List[BlastHit] sorted
                 by query_start position.  Pairs with no hit are absent.
        """
        if not self._check_blast_installed():
            raise RuntimeError("BLAST+ not found.")

        if word_size is None:
            word_size = 3 if self.seq_type == SeqType.PROTEIN else 11
        if threads is None:
            threads = max(1, multiprocessing.cpu_count() - 1)

        if verbose:
            print(f"  Creating BLAST database (multi-HSP mode, max_hsps={max_hsps})...")

        all_db = self.create_database(sequences, db_name='all_seqs_db_mhsp')

        query_path = Path(self.temp_dir) / 'queries_mhsp.fasta'
        with open(query_path, 'w') as f:
            for seq in sequences:
                f.write(f">{seq.id}\n{seq.seq}\n")

        n_seqs = len(sequences)
        cmd = [
            self.blast_cmd,
            '-query', str(query_path),
            '-db', str(all_db),
            '-gapopen', str(gap_open),
            '-gapextend', str(gap_extend),
            '-word_size', str(word_size),
            '-evalue', str(evalue),
            '-outfmt', '6 qseqid sseqid qstart qend sstart send qseq sseq evalue bitscore pident qlen slen',
            '-max_target_seqs', str(max(n_seqs + 5, 10)),
            '-max_hsps', str(max_hsps),
            '-num_threads', str(threads)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and 'error' in result.stderr.lower():
            raise RuntimeError(f"BLAST failed: {result.stderr}")

        # Collect ALL HSPs per pair
        multi_hits: Dict[Tuple[str, str], List[BlastHit]] = {}
        for hit in _parse_blast_output(result.stdout):
            if hit.query_id == hit.subject_id:
                continue
            key = (hit.query_id, hit.subject_id)
            multi_hits.setdefault(key, []).append(hit)

        # Sort each list by query_start
        for key in multi_hits:
            multi_hits[key].sort(key=lambda h: h.query_start)

        if verbose:
            n_pairs = len(multi_hits)
            n_multi = sum(1 for v in multi_hits.values() if len(v) > 1)
            print(f"  Found {n_pairs} pairwise alignments ({n_multi} with multiple HSPs)")

        return multi_hits

    def _retry_low_coverage_hits(self, hits: Dict[Tuple[str, str], BlastHit],
                                  sequences: List[Sequence],
                                  seq_lens: Dict[str, int],
                                  original_word_size: int,
                                  original_evalue: float,
                                  coverage_threshold: float,
                                  verbose: bool) -> Dict[Tuple[str, str], BlastHit]:
        """
        Retry BLAST for pairs where the alignment covers less than the threshold
        of the query sequence. Uses more sensitive parameters, multiple HSPs,
        and parallel execution via ProcessPoolExecutor.
        """
        seq_map = {s.id: s for s in sequences}
        low_coverage_pairs = set()

        for (qid, sid), hit in hits.items():
            qlen = seq_lens.get(qid, 0)
            if qlen == 0:
                continue
            coverage = hit.query_len_aligned / qlen
            if coverage < coverage_threshold:
                pair = tuple(sorted([qid, sid]))
                low_coverage_pairs.add(pair)

        if not low_coverage_pairs:
            return hits

        n_pairs = len(low_coverage_pairs)
        if verbose:
            print(f"  Coverage guard: {n_pairs} pairs below "
                  f"{coverage_threshold:.0%} coverage, retrying with sensitive parameters...")

        # Sensitive parameters
        sensitive_word_size = max(7, original_word_size - 4) if self.seq_type != SeqType.PROTEIN else max(2, original_word_size - 1)
        sensitive_evalue = max(original_evalue, 1e-3)
        sensitive_gap_open = 5 if self.seq_type != SeqType.PROTEIN else 11
        sensitive_gap_extend = 2 if self.seq_type != SeqType.PROTEIN else 1

        # Build args for parallel execution
        retry_args = []
        for id1, id2 in low_coverage_pairs:
            retry_args.append((
                id1, seq_map[id1].seq, id2, seq_map[id2].seq,
                self.blast_cmd, sensitive_gap_open, sensitive_gap_extend,
                sensitive_word_size, sensitive_evalue, self.temp_dir
            ))

        # Run retries in parallel
        n_workers = min(n_pairs, max(1, multiprocessing.cpu_count()))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_retry_single_pair, args): args
                       for args in retry_args}

            for future in as_completed(futures):
                (id1, id2), new_fwd, new_rev = future.result()

                for new_hit, key in [(new_fwd, (id1, id2)), (new_rev, (id2, id1))]:
                    if new_hit is None:
                        continue
                    qlen = seq_lens.get(new_hit.query_id, 0)
                    new_cov = new_hit.query_len_aligned / qlen if qlen > 0 else 0.0

                    old_hit = hits.get(key)
                    if old_hit is None:
                        hits[key] = new_hit
                        continue

                    old_cov = old_hit.query_len_aligned / seq_lens.get(old_hit.query_id, 1)
                    if new_cov > old_cov:
                        hits[key] = new_hit
                        if verbose:
                            print(f"    {key[0]} vs {key[1]}: coverage {old_cov:.0%} -> {new_cov:.0%}")

        return hits


    def run_vs_consensus(self, query: Sequence, consensus_seq: str,
                          gap_open: int = 5, gap_extend: int = 2,
                          word_size: int = 7, evalue: float = 1e-3) -> Optional[BlastHit]:
        """
        BLAST a single sequence against a consensus string.

        Uses sensitive parameters by default since this is for realigning
        divergent sequences during iterative refinement.
        """
        query_path = Path(self.temp_dir) / 'refine_query.fasta'
        subject_path = Path(self.temp_dir) / 'refine_consensus.fasta'

        with open(query_path, 'w') as f:
            f.write(f">{query.id}\n{query.seq}\n")
        with open(subject_path, 'w') as f:
            f.write(f">consensus\n{consensus_seq}\n")

        return _execute_blast(
            query_path, subject_path, self.blast_cmd,
            gap_open, gap_extend, word_size, evalue, max_hsps=3
        )


def compute_pairwise_scores(hits: Dict[Tuple[str, str], BlastHit],
                           seq_ids: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Compute symmetric pairwise score matrix from BLAST hits.
    
    Uses bitscore as the similarity measure.
    Returns dictionary mapping (id1, id2) to score (symmetric).
    """
    scores = {}
    
    for id1 in seq_ids:
        for id2 in seq_ids:
            if id1 == id2:
                continue
            
            # Check both directions
            score = 0.0
            if (id1, id2) in hits:
                score = max(score, hits[(id1, id2)].bitscore)
            if (id2, id1) in hits:
                score = max(score, hits[(id2, id1)].bitscore)
            
            # Store symmetrically
            scores[(id1, id2)] = score
            scores[(id2, id1)] = score
    
    return scores


def find_center_sequence(scores: Dict[Tuple[str, str], float],
                        seq_ids: List[str]) -> str:
    """
    Find the center sequence (highest total score to all others).
    
    This will be the reference for center-star alignment.
    """
    total_scores = {}
    
    for seq_id in seq_ids:
        total = sum(scores.get((seq_id, other), 0.0) 
                   for other in seq_ids if other != seq_id)
        total_scores[seq_id] = total
    
    # Return ID with highest total score
    return max(total_scores.keys(), key=lambda x: total_scores[x])
