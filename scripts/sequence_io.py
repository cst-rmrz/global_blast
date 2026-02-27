"""
Sequence I/O utilities for BLAST MSA

Handles:
- FASTA parsing
- Automatic sequence type detection (protein vs nucleotide)
- Multiple output formats (FASTA, Phylip, Clustal, Nexus)
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterator
from enum import Enum, auto


class SeqType(Enum):
    """Sequence type enumeration"""
    PROTEIN = auto()
    NUCLEOTIDE = auto()
    UNKNOWN = auto()


@dataclass
class Sequence:
    """Simple sequence container"""
    id: str
    description: str
    seq: str
    
    @property
    def full_header(self) -> str:
        if self.description:
            return f"{self.id} {self.description}"
        return self.id
    
    def __len__(self) -> int:
        return len(self.seq)


@dataclass
class Alignment:
    """Multiple sequence alignment container"""
    sequences: List[Sequence]
    seq_type: SeqType
    parameters: Optional[Dict] = None
    score: Optional[float] = None
    
    @property
    def n_seqs(self) -> int:
        return len(self.sequences)
    
    @property
    def length(self) -> int:
        if not self.sequences:
            return 0
        return len(self.sequences[0].seq)
    
    def is_valid(self) -> bool:
        """Check that all sequences have same length"""
        if not self.sequences:
            return False
        lengths = set(len(s.seq) for s in self.sequences)
        return len(lengths) == 1


def parse_fasta(filepath: Path) -> List[Sequence]:
    """
    Parse a FASTA file into a list of Sequence objects.
    
    Handles:
    - Standard FASTA format
    - Multi-line sequences
    - Various line endings
    """
    sequences = []
    current_id = None
    current_desc = ""
    current_seq_parts = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences.append(Sequence(
                        id=current_id,
                        description=current_desc,
                        seq=''.join(current_seq_parts).upper()
                    ))
                
                # Parse new header
                header = line[1:].strip()
                parts = header.split(None, 1)  # Split on first whitespace
                current_id = parts[0] if parts else "unnamed"
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq_parts = []
            
            elif line and not line.startswith(';'):  # Skip empty lines and comments
                # Remove any whitespace/numbers that might be in sequence
                clean_seq = re.sub(r'[\s\d]', '', line)
                current_seq_parts.append(clean_seq)
    
    # Don't forget the last sequence
    if current_id is not None:
        sequences.append(Sequence(
            id=current_id,
            description=current_desc,
            seq=''.join(current_seq_parts).upper()
        ))
    
    return sequences


def detect_sequence_type(sequences: List[Sequence], sample_size: int = 1000) -> SeqType:
    """
    Automatically detect whether sequences are protein or nucleotide.
    
    Strategy:
    - Sample characters from sequences
    - If >80% are ACGTU (allowing N for ambiguous), likely nucleotide
    - Otherwise, likely protein
    
    Also checks for amino acid-specific characters (EFILPQZ) which
    are definitive for protein.
    """
    if not sequences:
        return SeqType.UNKNOWN
    
    # Collect sample of characters
    all_chars = []
    for seq in sequences:
        all_chars.extend(list(seq.seq[:sample_size // len(sequences) + 1]))
        if len(all_chars) >= sample_size:
            break
    
    if not all_chars:
        return SeqType.UNKNOWN
    
    # Remove gaps and ambiguous characters for analysis
    chars = [c for c in all_chars if c not in '-.*X']
    
    if not chars:
        return SeqType.UNKNOWN
    
    # Characters that are ONLY valid for protein (not nucleotide)
    protein_only = set('EFIJLOPQZ')
    
    # Check for definitive protein characters
    char_set = set(chars)
    if char_set & protein_only:
        return SeqType.PROTEIN
    
    # Count nucleotide characters
    nucleotide_chars = set('ACGTUN')
    n_nucleotide = sum(1 for c in chars if c in nucleotide_chars)
    
    # If >85% nucleotide characters, call it nucleotide
    if n_nucleotide / len(chars) > 0.85:
        return SeqType.NUCLEOTIDE
    
    return SeqType.PROTEIN


def write_fasta(alignment: Alignment, filepath: Path, wrap_width: int = 80) -> None:
    """Write alignment in FASTA format"""
    with open(filepath, 'w') as f:
        for seq in alignment.sequences:
            f.write(f">{seq.full_header}\n")
            
            if wrap_width > 0:
                for i in range(0, len(seq.seq), wrap_width):
                    f.write(seq.seq[i:i+wrap_width] + '\n')
            else:
                f.write(seq.seq + '\n')


def write_phylip(alignment: Alignment, filepath: Path, relaxed: bool = False) -> None:
    """
    Write alignment in Phylip format.
    
    Args:
        relaxed: If True, use relaxed Phylip (longer names allowed)
                 If False, use strict Phylip (names truncated to 10 chars)
    """
    with open(filepath, 'w') as f:
        # Header line: number of sequences and alignment length
        f.write(f" {alignment.n_seqs} {alignment.length}\n")
        
        for seq in alignment.sequences:
            if relaxed:
                # Relaxed format: name followed by two spaces, then sequence
                name = seq.id.replace(' ', '_')
                f.write(f"{name}  {seq.seq}\n")
            else:
                # Strict format: name padded/truncated to exactly 10 characters
                name = seq.id[:10].ljust(10)
                f.write(f"{name}{seq.seq}\n")


def write_clustal(alignment: Alignment, filepath: Path, wrap_width: int = 60) -> None:
    """Write alignment in Clustal format"""
    # Find the longest sequence name for padding
    max_name_len = max(len(s.id) for s in alignment.sequences)
    max_name_len = min(max_name_len, 30)  # Cap at 30 characters
    
    with open(filepath, 'w') as f:
        f.write("CLUSTAL W (1.83) multiple sequence alignment\n")
        f.write("           (generated by BLAST MSA)\n\n")
        
        # Write alignment in blocks
        for block_start in range(0, alignment.length, wrap_width):
            block_end = min(block_start + wrap_width, alignment.length)
            
            for seq in alignment.sequences:
                name = seq.id[:max_name_len].ljust(max_name_len)
                block = seq.seq[block_start:block_end]
                f.write(f"{name} {block}\n")
            
            # Conservation line (simplified - just spacing)
            f.write(" " * (max_name_len + 1))
            
            # Calculate conservation for this block
            for i in range(block_start, block_end):
                col = [s.seq[i] for s in alignment.sequences]
                if len(set(col)) == 1 and col[0] != '-':
                    f.write('*')
                elif all(c != '-' for c in col):
                    f.write('.')
                else:
                    f.write(' ')
            f.write('\n\n')


def write_nexus(alignment: Alignment, filepath: Path) -> None:
    """Write alignment in NEXUS format"""
    datatype = "protein" if alignment.seq_type == SeqType.PROTEIN else "dna"
    
    with open(filepath, 'w') as f:
        f.write("#NEXUS\n\n")
        f.write("BEGIN DATA;\n")
        f.write(f"  DIMENSIONS NTAX={alignment.n_seqs} NCHAR={alignment.length};\n")
        f.write(f"  FORMAT DATATYPE={datatype} MISSING=? GAP=-;\n")
        f.write("  MATRIX\n")
        
        # Find longest name for padding
        max_name_len = max(len(s.id) for s in alignment.sequences)
        
        for seq in alignment.sequences:
            name = seq.id.replace(' ', '_').ljust(max_name_len)
            f.write(f"    {name}  {seq.seq}\n")
        
        f.write("  ;\n")
        f.write("END;\n")


def write_alignment(alignment: Alignment, filepath: Path, 
                   format: str = "fasta", wrap_width: int = 80) -> None:
    """
    Write alignment to file in specified format.
    
    Args:
        alignment: Alignment object to write
        filepath: Output file path
        format: One of 'fasta', 'phylip', 'phylip-relaxed', 'clustal', 'nexus'
        wrap_width: Line width for wrapping (where applicable)
    """
    filepath = Path(filepath)
    format = format.lower()
    
    if format == 'fasta':
        write_fasta(alignment, filepath, wrap_width)
    elif format == 'phylip':
        write_phylip(alignment, filepath, relaxed=False)
    elif format == 'phylip-relaxed':
        write_phylip(alignment, filepath, relaxed=True)
    elif format == 'clustal':
        write_clustal(alignment, filepath, wrap_width=60)
    elif format == 'nexus':
        write_nexus(alignment, filepath)
    else:
        raise ValueError(f"Unknown format: {format}. "
                        f"Supported: fasta, phylip, phylip-relaxed, clustal, nexus")


def format_from_extension(filepath: Path) -> str:
    """Infer output format from file extension"""
    ext = filepath.suffix.lower()
    mapping = {
        '.fasta': 'fasta',
        '.fa': 'fasta',
        '.fna': 'fasta',
        '.faa': 'fasta',
        '.phy': 'phylip',
        '.phylip': 'phylip',
        '.aln': 'clustal',
        '.clustal': 'clustal',
        '.nex': 'nexus',
        '.nexus': 'nexus',
        '.nxs': 'nexus',
    }
    return mapping.get(ext, 'fasta')
