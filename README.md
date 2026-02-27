# BLAST MSA

Multiple Sequence Alignment using the BLAST algorithm.

## Why BLAST for MSA?

Traditional MSA tools like ClustalW and MUSCLE use global alignment algorithms that can struggle with:
- Divergent sequences
- Domain shuffling
- Sequences with conserved local regions but variable termini

BLAST's local alignment algorithm often finds biologically meaningful alignments that global aligners miss. This tool leverages BLAST's superior local alignment to build multiple sequence alignments.

## Requirements

- Python 3.7+
- NCBI BLAST+ tools

### Installing BLAST+

**Ubuntu/Debian:**
```bash
sudo apt install ncbi-blast+
```

**macOS (Homebrew):**
```bash
brew install blast
```

**Manual download:**
https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

## Usage

### Basic usage

```bash
# Align sequences, output as FASTA
python blast_msa.py sequences.fasta -o aligned.fasta

# Output as Phylip format
python blast_msa.py sequences.fasta -o aligned.phy

# Output as Clustal format with verbose output
python blast_msa.py sequences.fasta -o aligned.aln -f clustal -v
```

### Parameter optimization

The `--optimize` flag performs a grid search over gap penalties and word sizes to find the best alignment:

```bash
# Optimize using sum-of-pairs score (default)
python blast_msa.py sequences.fasta -o aligned.fasta --optimize

# Optimize using percent identity
python blast_msa.py sequences.fasta -o aligned.fasta --optimize --metric percent_identity

# Optimize using column conservation score
python blast_msa.py sequences.fasta -o aligned.fasta --optimize --metric column_score
```

### Custom parameters

```bash
# Set specific BLAST parameters
python blast_msa.py sequences.fasta -o aligned.fasta --gap-open 9 --gap-extend 2

# Use a custom parameters file
python blast_msa.py sequences.fasta -o aligned.fasta --params my_params.params
```

### Output formats

| Format | Extension | Description |
|--------|-----------|-------------|
| FASTA | .fasta, .fa, .fna, .faa | Standard FASTA alignment |
| Phylip | .phy, .phylip | Strict Phylip (10-char names) |
| Phylip-relaxed | - | Relaxed Phylip (longer names) |
| Clustal | .aln, .clustal | Clustal format with conservation |
| NEXUS | .nex, .nexus, .nxs | NEXUS format for phylogenetics |

## Configuration

Default parameters are stored in `blast.params`. Edit this file to change defaults:

```ini
# BLAST Algorithm Parameters
gap_open = 11
gap_extend = 1
word_size_protein = 3
word_size_nucleotide = 11
evalue = 1e-5

# Optimization Parameters
optimize_gap_open = 5,7,9,11,13,15
optimize_gap_extend = 1,2,3
optimize_metric = sp_score

# Output Parameters
default_format = fasta
wrap_width = 80
```

## Algorithm

BLAST MSA uses the **center-star algorithm**:

1. **Pairwise BLAST**: Run BLAST between all sequence pairs
2. **Find center**: Identify the sequence with highest total BLAST score to all others
3. **Align to center**: Extend each pairwise alignment to full sequence length
4. **Merge**: Combine pairwise alignments by inserting gaps to maintain consistency

### Terminal extension

BLAST produces local alignments. This tool extends alignments to sequence termini:
- Unaligned N-terminal regions are left-padded with gaps
- Unaligned C-terminal regions are right-padded with gaps
- Terminal gaps are free (not penalized in scoring)

## Scoring metrics

For optimization, three metrics are available:

| Metric | Description | Best for |
|--------|-------------|----------|
| `sp_score` | Sum-of-pairs score | General purpose |
| `percent_identity` | Average pairwise identity | Maximizing similarity |
| `column_score` | Column conservation score | Maximizing conservation |

## Sequence type detection

The tool automatically detects whether sequences are protein or nucleotide:
- If sequences contain amino acid-specific characters (E, F, I, L, P, Q, Z), they're protein
- If >85% of characters are nucleotides (A, C, G, T, U, N), they're nucleotide
- Otherwise, they're treated as protein

## Examples

### Aligning tomato defense genes

```bash
python blast_msa.py defense_genes.fasta -o defense_aligned.phy --optimize -v
```

### Aligning fungal effector proteins

```bash
python blast_msa.py effectors.faa -o effectors.aln -f clustal --optimize --metric percent_identity
```

### Quick alignment with default parameters

```bash
python blast_msa.py my_sequences.fasta -o aligned.fasta
```

## Troubleshooting

**"BLAST+ not found"**: Install NCBI BLAST+ tools (see Requirements section)

**"Need at least 2 sequences"**: Input file must contain multiple sequences

**Poor alignment quality**: Try `--optimize` flag or adjust parameters in `blast.params`

**Very divergent sequences**: BLAST may not find hits. Consider lowering E-value threshold or using `--evalue 10`

## License

MIT License - feel free to modify and distribute.

## Author

Created for plant pathology research on tomato-*Alternaria* interactions.
