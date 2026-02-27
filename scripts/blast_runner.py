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
        
        # Build BLAST command
        # Output format 6: tabular with specific columns
        # qseqid sseqid qstart qend sstart send qseq sseq evalue bitscore pident
        cmd = [
            self.blast_cmd,
            '-query', str(query_path),
            '-subject', str(subject_path),
            '-gapopen', str(gap_open),
            '-gapextend', str(gap_extend),
            '-word_size', str(word_size),
            '-evalue', str(evalue),
            '-outfmt', '6 qseqid sseqid qstart qend sstart send qseq sseq evalue bitscore pident',
            '-max_target_seqs', '1',
            '-max_hsps', '1'  # Only best HSP
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # BLAST can return non-zero for no hits, check stderr
            if 'BLAST Database error' in result.stderr:
                raise RuntimeError(f"BLAST error: {result.stderr}")
            return None
        
        # Parse output
        output = result.stdout.strip()
        if not output:
            return None
        
        # Take first (best) hit
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
    
    def run_all_pairwise(self, sequences: List[Sequence],
                         gap_open: int = 11, gap_extend: int = 1,
                         word_size: int = None, evalue: float = 1e-5,
                         verbose: bool = False) -> Dict[Tuple[str, str], BlastHit]:
        """
        Run all pairwise BLASTs between sequences.
        
        Returns a dictionary mapping (query_id, subject_id) tuples to BlastHit objects.
        """
        hits = {}
        n_seqs = len(sequences)
        total_pairs = n_seqs * (n_seqs - 1) // 2
        
        pair_count = 0
        for i, seq1 in enumerate(sequences):
            for j, seq2 in enumerate(sequences):
                if i >= j:
                    continue
                
                pair_count += 1
                if verbose:
                    print(f"\r  BLAST pair {pair_count}/{total_pairs}: {seq1.id} vs {seq2.id}",
                          end='', flush=True)
                
                # Run BLAST in both directions and keep best
                hit_forward = self.run_pairwise(
                    seq1, seq2, gap_open, gap_extend, word_size, evalue
                )
                hit_reverse = self.run_pairwise(
                    seq2, seq1, gap_open, gap_extend, word_size, evalue
                )
                
                # Store both directions if found
                if hit_forward:
                    hits[(seq1.id, seq2.id)] = hit_forward
                if hit_reverse:
                    hits[(seq2.id, seq1.id)] = hit_reverse
        
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
