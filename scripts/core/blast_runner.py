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
                   evalue: float) -> Optional[BlastHit]:
    """Execute a single BLAST command and parse result."""
    cmd = [
        blast_cmd,
        '-query', str(query_path),
        '-subject', str(subject_path),
        '-gapopen', str(gap_open),
        '-gapextend', str(gap_extend),
        '-word_size', str(word_size),
        '-evalue', str(evalue),
        '-outfmt', '6 qseqid sseqid qstart qend sstart send qseq sseq evalue bitscore pident',
        '-max_target_seqs', '1',
        '-max_hsps', '1'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return None
    
    output = result.stdout.strip()
    if not output:
        return None
    
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
    
    def create_database(self, sequences: List[Sequence]) -> Path:
        """Create a BLAST database from sequences"""
        if not self._check_blast_installed():
            raise RuntimeError(
                "BLAST+ not found. Please install NCBI BLAST+ tools.\n"
                "Ubuntu/Debian: sudo apt install ncbi-blast+\n"
                "macOS: brew install blast\n"
                "Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
            )
        
        # Write sequences to temp FASTA
        fasta_path = Path(self.temp_dir) / 'sequences.fasta'
        with open(fasta_path, 'w') as f:
            for seq in sequences:
                f.write(f">{seq.id}\n{seq.seq}\n")
        
        # Create BLAST database
        self.db_path = Path(self.temp_dir) / 'blastdb'
        
        cmd = [
            'makeblastdb',
            '-in', str(fasta_path),
            '-dbtype', self.makedb_type,
            '-out', str(self.db_path),
            '-parse_seqids'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"makeblastdb failed: {result.stderr}")
        
        return self.db_path
    
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
                         threads: int = None) -> Dict[Tuple[str, str], BlastHit]:
        """
        Run all pairwise BLASTs between sequences in parallel.
        
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
        
        # Build list of all pairs to process
        pairs = []
        n_seqs = len(sequences)
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                pairs.append((
                    sequences[i].id, sequences[i].seq,
                    sequences[j].id, sequences[j].seq,
                    self.blast_cmd, gap_open, gap_extend, word_size, evalue,
                    self.temp_dir
                ))
        
        total_pairs = len(pairs)
        
        if verbose:
            print(f"  Running {total_pairs} pairwise BLASTs using {threads} threads...")
        
        hits = {}
        completed = 0
        
        # Run in parallel
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(_run_single_blast, pair): pair for pair in pairs}
            
            for future in as_completed(futures):
                completed += 1
                if verbose:
                    print(f"\r  Completed {completed}/{total_pairs} pairs", end='', flush=True)
                
                try:
                    (id1, id2), hit_forward, hit_reverse = future.result()
                    
                    if hit_forward:
                        hits[(id1, id2)] = hit_forward
                    if hit_reverse:
                        hits[(id2, id1)] = hit_reverse
                        
                except Exception as e:
                    if verbose:
                        print(f"\n  Warning: pair failed: {e}")
        
        if verbose:
            print()  # Newline after progress
        
        return hits


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
