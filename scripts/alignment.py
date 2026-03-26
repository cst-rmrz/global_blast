"""
Alignment construction module for BLAST MSA

Implements center-star algorithm to build MSA from pairwise BLAST alignments.
Handles terminal extension to create full-length alignments.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

from .sequence_io import Sequence, Alignment, SeqType
from .blast_runner import BlastHit, BlastRunner


@dataclass
class AlignedPair:
    """
    Aligned pair of sequences with full-length alignment.
    
    Both sequences are the same length (with gaps).
    Includes original (ungapped) sequences for reference.
    """
    seq1_id: str
    seq2_id: str
    seq1_aligned: str
    seq2_aligned: str
    seq1_original: str
    seq2_original: str


def extend_pairwise_alignment(hit: BlastHit, 
                              query_seq: str, 
                              subject_seq: str) -> AlignedPair:
    """
    Extend a local BLAST alignment to cover full sequence lengths.
    
    Strategy:
    1. Add unaligned N-terminal regions (with gaps in the other sequence)
    2. Keep the BLAST-aligned middle region
    3. Add unaligned C-terminal regions (with gaps in the other sequence)
    
    Terminal gaps are "free" in typical MSA scoring, so this doesn't
    penalize sequences for length differences.
    """
    # Get the aligned portions from BLAST
    query_aln = hit.query_seq
    subject_aln = hit.subject_seq
    
    # N-terminal extensions
    # Query starts at hit.query_start (1-based), so we need positions 0 to query_start-2
    query_n_term = query_seq[:hit.query_start - 1]
    subject_n_term = subject_seq[:hit.subject_start - 1]
    
    # C-terminal extensions  
    # Query ends at hit.query_end (1-based), so we need positions query_end to end
    query_c_term = query_seq[hit.query_end:]
    subject_c_term = subject_seq[hit.subject_end:]
    
    # Build extended alignment
    # N-terminal: align the overhangs
    n_term_query, n_term_subject = _extend_terminal(
        query_n_term, subject_n_term, is_n_terminal=True
    )
    
    # C-terminal: align the overhangs
    c_term_query, c_term_subject = _extend_terminal(
        query_c_term, subject_c_term, is_n_terminal=False
    )
    
    # Concatenate all parts
    full_query = n_term_query + query_aln + c_term_query
    full_subject = n_term_subject + subject_aln + c_term_subject
    
    # Sanity check
    assert len(full_query) == len(full_subject), \
        f"Alignment length mismatch: {len(full_query)} vs {len(full_subject)}"
    
    return AlignedPair(
        seq1_id=hit.query_id,
        seq2_id=hit.subject_id,
        seq1_aligned=full_query,
        seq2_aligned=full_subject,
        seq1_original=query_seq,
        seq2_original=subject_seq
    )


def _extend_terminal(seq1_term: str, seq2_term: str, 
                     is_n_terminal: bool) -> Tuple[str, str]:
    """
    Extend terminal overhangs with gap padding.
    
    For N-terminal: longer sequence gets right-aligned (gaps on left of shorter)
    For C-terminal: longer sequence gets left-aligned (gaps on right of shorter)
    """
    len1 = len(seq1_term)
    len2 = len(seq2_term)
    
    if len1 == 0 and len2 == 0:
        return "", ""
    
    if len1 >= len2:
        # seq1 is longer or equal
        gap_len = len1 - len2
        if is_n_terminal:
            # Gaps go at the start of seq2
            return seq1_term, '-' * gap_len + seq2_term
        else:
            # Gaps go at the end of seq2
            return seq1_term, seq2_term + '-' * gap_len
    else:
        # seq2 is longer
        gap_len = len2 - len1
        if is_n_terminal:
            # Gaps go at the start of seq1
            return '-' * gap_len + seq1_term, seq2_term
        else:
            # Gaps go at the end of seq1
            return seq1_term + '-' * gap_len, seq2_term


def create_alignment_no_hit(seq1: Sequence, seq2: Sequence) -> AlignedPair:
    """
    Create an alignment when no BLAST hit was found.
    
    Simply concatenates sequences end-to-end with gaps.
    This is a fallback for very divergent sequences.
    """
    len1 = len(seq1.seq)
    len2 = len(seq2.seq)
    
    # Put seq1 first, then seq2 (with gaps filling the other)
    seq1_aligned = seq1.seq + '-' * len2
    seq2_aligned = '-' * len1 + seq2.seq
    
    return AlignedPair(
        seq1_id=seq1.id,
        seq2_id=seq2.id,
        seq1_aligned=seq1_aligned,
        seq2_aligned=seq2_aligned,
        seq1_original=seq1.seq,
        seq2_original=seq2.seq
    )


class CenterStarAligner:
    """
    Builds MSA using center-star algorithm.
    
    1. Identify center sequence (highest total pairwise score)
    2. Align all other sequences to center
    3. Merge alignments by inserting gaps to maintain consistency
    """
    
    def __init__(self, sequences: List[Sequence], seq_type: SeqType):
        self.sequences = {s.id: s for s in sequences}
        self.seq_ids = [s.id for s in sequences]
        self.seq_type = seq_type
        self.center_id: Optional[str] = None
    
    def build_msa(self, hits: Dict[Tuple[str, str], BlastHit],
                  center_id: Optional[str] = None,
                  verbose: bool = False) -> Alignment:
        """
        Build MSA from pairwise BLAST hits.
        
        Args:
            hits: Dictionary of pairwise BLAST results
            center_id: Optional center sequence ID (auto-detected if None)
            verbose: Print progress information
        
        Returns:
            Alignment object with all sequences aligned
        """
        if len(self.sequences) < 2:
            # Single sequence - just return it
            seq = list(self.sequences.values())[0]
            return Alignment(
                sequences=[seq],
                seq_type=self.seq_type
            )
        
        # Use provided center or find it
        if center_id is None:
            center_id = self._find_center(hits)
        self.center_id = center_id
        
        if verbose:
            print(f"  Center sequence: {center_id}")
        
        center_seq = self.sequences[center_id]
        other_ids = [sid for sid in self.seq_ids if sid != center_id]
        
        # Get pairwise alignments of all sequences to center
        pairwise_alignments = {}
        
        for other_id in other_ids:
            other_seq = self.sequences[other_id]
            
            # Try to find a hit (check both directions)
            hit = hits.get((center_id, other_id)) or hits.get((other_id, center_id))
            
            if hit:
                # Ensure hit is oriented with center as query
                if hit.query_id != center_id:
                    # Swap the hit orientation
                    hit = BlastHit(
                        query_id=hit.subject_id,
                        subject_id=hit.query_id,
                        query_start=hit.subject_start,
                        query_end=hit.subject_end,
                        subject_start=hit.query_start,
                        subject_end=hit.query_end,
                        query_seq=hit.subject_seq,
                        subject_seq=hit.query_seq,
                        evalue=hit.evalue,
                        bitscore=hit.bitscore,
                        identity=hit.identity
                    )
                
                pair = extend_pairwise_alignment(hit, center_seq.seq, other_seq.seq)
            else:
                # No hit found - create fallback alignment
                pair = create_alignment_no_hit(center_seq, other_seq)
            
            pairwise_alignments[other_id] = pair
        
        # Merge all pairwise alignments
        aligned_seqs = self._merge_alignments(center_seq, pairwise_alignments, verbose)
        
        return Alignment(
            sequences=aligned_seqs,
            seq_type=self.seq_type
        )
    
    def _find_center(self, hits: Dict[Tuple[str, str], BlastHit]) -> str:
        """Find center sequence with highest total bitscore to others"""
        scores = {}
        
        for seq_id in self.seq_ids:
            total = 0.0
            for other_id in self.seq_ids:
                if other_id == seq_id:
                    continue
                
                # Check both directions
                hit1 = hits.get((seq_id, other_id))
                hit2 = hits.get((other_id, seq_id))
                
                if hit1:
                    total += hit1.bitscore
                if hit2:
                    total += hit2.bitscore
            
            scores[seq_id] = total
        
        return max(scores.keys(), key=lambda x: scores[x])
    
    def _merge_alignments(self, center_seq: Sequence,
                          pairwise: Dict[str, AlignedPair],
                          verbose: bool = False) -> List[Sequence]:
        """
        Merge pairwise alignments into a single MSA.
        
        The key insight: gaps in the center sequence from different pairwise
        alignments may occur at different positions. We need to insert additional
        gaps to make all center sequences identical, then propagate those gaps
        to the corresponding other sequences.
        """
        if not pairwise:
            return [center_seq]
        
        # Build a "master" center sequence with all required gaps
        # Track gap positions from each pairwise alignment
        
        # First pass: collect all gap insertion points needed in the center
        # We'll work through the ungapped center and note where gaps appear
        # in each pairwise alignment
        
        all_center_aligned = [pair.seq1_aligned for pair in pairwise.values()]
        
        # Use the first aligned center as starting point
        master_center = list(all_center_aligned[0])
        master_mapping = list(range(len(master_center)))  # Maps master positions to original
        
        # For each other alignment, find gaps that need to be inserted into master
        for i, pair in enumerate(pairwise.values()):
            if i == 0:
                continue
            
            center_aln = pair.seq1_aligned
            
            # Align the two center versions to find discrepancies
            master_center, master_mapping = self._reconcile_center(
                master_center, master_mapping, center_aln
            )
        
        # Now master_center has all gaps from all pairwise alignments
        # Rebuild each other sequence with matching gaps
        
        aligned_sequences = []
        
        # Add center sequence
        aligned_sequences.append(Sequence(
            id=center_seq.id,
            description=center_seq.description,
            seq=''.join(master_center)
        ))
        
        # Add other sequences with appropriate gaps
        for other_id, pair in pairwise.items():
            other_aligned = self._apply_master_gaps(
                pair.seq1_aligned,  # center from this pair
                pair.seq2_aligned,  # other from this pair
                master_center,
                master_mapping
            )
            
            other_seq = self.sequences[other_id]
            aligned_sequences.append(Sequence(
                id=other_seq.id,
                description=other_seq.description,
                seq=other_aligned
            ))
        
        return aligned_sequences
    
    def _reconcile_center(self, master: List[str], master_map: List[int],
                          new_center: str) -> Tuple[List[str], List[int]]:
        """
        Reconcile a new center alignment with the master.
        
        Inserts gaps into master where new_center has gaps that master doesn't.
        Returns updated master and mapping.
        """
        result = []
        result_map = []
        
        master_idx = 0
        new_idx = 0
        
        while master_idx < len(master) or new_idx < len(new_center):
            if master_idx >= len(master):
                # Master exhausted, add from new
                result.append(new_center[new_idx])
                result_map.append(-1)  # No mapping for this gap
                new_idx += 1
            elif new_idx >= len(new_center):
                # New exhausted, add from master
                result.append(master[master_idx])
                result_map.append(master_map[master_idx])
                master_idx += 1
            elif master[master_idx] == '-' and new_center[new_idx] == '-':
                # Both have gaps
                result.append('-')
                result_map.append(-1)
                master_idx += 1
                new_idx += 1
            elif master[master_idx] == '-':
                # Master has gap, new doesn't - keep master's gap
                result.append('-')
                result_map.append(master_map[master_idx])
                master_idx += 1
            elif new_center[new_idx] == '-':
                # New has gap, master doesn't - insert gap into master
                result.append('-')
                result_map.append(-1)
                new_idx += 1
            else:
                # Both have characters - should match
                result.append(master[master_idx])
                result_map.append(master_map[master_idx])
                master_idx += 1
                new_idx += 1
        
        return result, result_map
    
    def _apply_master_gaps(self, pair_center: str, pair_other: str,
                           master: List[str], master_map: List[int]) -> str:
        """
        Apply the master gap structure to a pairwise alignment.
        
        Returns the 'other' sequence with gaps inserted to match master structure.
        """
        result = []
        
        pair_idx = 0
        master_idx = 0
        
        while master_idx < len(master):
            if pair_idx < len(pair_center):
                # Check if we need to insert extra gaps
                if master[master_idx] == '-':
                    # Master has gap at this position
                    if pair_center[pair_idx] == '-':
                        # Pair also has gap - use pair's other character
                        result.append(pair_other[pair_idx])
                        pair_idx += 1
                    else:
                        # Pair doesn't have gap here - insert gap in result
                        result.append('-')
                else:
                    # Master has character
                    result.append(pair_other[pair_idx])
                    pair_idx += 1
            else:
                # Pair exhausted but master continues - add gaps
                result.append('-')
            
            master_idx += 1
        
        return ''.join(result)

    def refine_msa(self, alignment: Alignment,
                   max_iterations: int = 3,
                   verbose: bool = False) -> Alignment:
        """
        Iterative refinement: detect poorly-aligned sequences, remove them,
        build consensus from remaining, re-BLAST against consensus, reinsert.

        Repeats until no sequences are flagged or max_iterations reached.
        """
        import numpy as np

        current = alignment

        for iteration in range(max_iterations):
            scores = compute_per_sequence_identity(current)
            if not scores:
                break

            values = list(scores.values())
            mean_score = np.mean(values)
            std_score = np.std(values)

            # Flag sequences below mean - 1.5 * std
            threshold = mean_score - 1.5 * std_score
            poor_ids = [sid for sid, score in scores.items()
                        if score < threshold and score < mean_score]

            if not poor_ids:
                if verbose:
                    print(f"  Refinement iteration {iteration + 1}: "
                          f"no poorly-aligned sequences detected")
                break

            if verbose:
                print(f"  Refinement iteration {iteration + 1}: "
                      f"realigning {len(poor_ids)} sequences "
                      f"(threshold: {threshold:.1f}% identity)")
                for sid in poor_ids:
                    print(f"    {sid}: {scores[sid]:.1f}%")

            # Build consensus from the good sequences
            good_seqs = [s for s in current.sequences if s.id not in poor_ids]
            good_alignment = Alignment(sequences=good_seqs, seq_type=current.seq_type)
            consensus_seq = build_consensus(good_alignment, gap_threshold=0.5)

            if not consensus_seq:
                if verbose:
                    print(f"  Could not build consensus, stopping refinement")
                break

            # Re-BLAST poor sequences against consensus
            poor_seq_map = {s.id: s for s in self.sequences.values()
                           if s.id in poor_ids}

            # Sensitive parameters for realignment
            sensitive_ws = 7 if self.seq_type != SeqType.PROTEIN else 2
            sensitive_evalue = 1e-3

            new_hits = {}
            with BlastRunner(self.seq_type) as runner:
                for sid, seq in poor_seq_map.items():
                    hit = runner.run_vs_consensus(
                        seq, consensus_seq,
                        gap_open=5 if self.seq_type != SeqType.PROTEIN else 11,
                        gap_extend=2 if self.seq_type != SeqType.PROTEIN else 1,
                        word_size=sensitive_ws,
                        evalue=sensitive_evalue
                    )
                    if hit:
                        # The hit has subject_id='consensus', remap to center
                        new_hits[(self.center_id, sid)] = BlastHit(
                            query_id=hit.subject_id,
                            subject_id=hit.query_id,
                            query_start=hit.subject_start,
                            query_end=hit.subject_end,
                            subject_start=hit.query_start,
                            subject_end=hit.query_end,
                            query_seq=hit.subject_seq,
                            subject_seq=hit.query_seq,
                            evalue=hit.evalue,
                            bitscore=hit.bitscore,
                            identity=hit.identity
                        )

            if not new_hits:
                if verbose:
                    print(f"  No improved hits found, stopping refinement")
                break

            # Rebuild MSA: run full all-vs-all BLAST but replace hits for
            # the poor sequences with the consensus-based hits
            # Simpler approach: rebuild from the existing good alignment
            # by re-extending the poor sequences with new hits

            # Get the consensus as a Sequence for alignment extension
            consensus_as_seq = Sequence(
                id='_consensus_',
                description='consensus',
                seq=consensus_seq
            )

            # Build new pairwise alignments for poor sequences against consensus
            refined_seqs = list(good_seqs)  # Start with good sequences as-is

            for sid in poor_ids:
                hit_key = (self.center_id, sid)
                if hit_key in new_hits:
                    hit = new_hits[hit_key]
                    # Extend the pairwise alignment (consensus as query, poor seq as subject)
                    pair = extend_pairwise_alignment(
                        hit, consensus_seq, poor_seq_map[sid].seq
                    )
                    # The aligned subject is our refined sequence, but it's aligned
                    # to the consensus which has the same coordinate space as the
                    # good alignment. We need to insert it at the right position.

                    # Map the consensus-aligned sequence back to the MSA columns
                    aligned_seq = self._map_to_msa_columns(
                        pair.seq2_aligned, pair.seq1_aligned, good_alignment
                    )

                    refined_seqs.append(Sequence(
                        id=sid,
                        description=poor_seq_map[sid].description,
                        seq=aligned_seq
                    ))
                else:
                    # Keep original if no new hit
                    orig = next((s for s in current.sequences if s.id == sid), None)
                    if orig:
                        refined_seqs.append(orig)

            # Validate all same length - pad with gaps if needed
            max_len = max(len(s.seq) for s in refined_seqs)
            for s in refined_seqs:
                if len(s.seq) < max_len:
                    s.seq = s.seq + '-' * (max_len - len(s.seq))

            current = Alignment(
                sequences=refined_seqs,
                seq_type=current.seq_type,
                parameters=current.parameters,
                score=current.score
            )

        return current

    def _map_to_msa_columns(self, seq_aligned_to_consensus: str,
                             consensus_aligned: str,
                             good_alignment: Alignment) -> str:
        """
        Map a sequence aligned to the consensus back into the MSA column space.

        The consensus was built by stripping gap-heavy columns from the MSA.
        We need to reverse that mapping: for each MSA column, determine whether
        it contributed to the consensus, and if so, take the corresponding
        character from the realigned sequence.
        """
        # Figure out which MSA columns contributed to the consensus
        # (same logic as build_consensus: columns where gap fraction <= 0.5)
        sequences = [s.seq for s in good_alignment.sequences]
        n_seqs = len(sequences)

        # Build mapping: msa_col -> consensus_position (or -1 if skipped)
        consensus_pos = 0
        msa_to_consensus = []
        for col in range(good_alignment.length):
            column = [seq[col] for seq in sequences]
            gap_fraction = column.count('-') / n_seqs
            if gap_fraction > 0.5:
                msa_to_consensus.append(-1)  # This column was skipped
            else:
                msa_to_consensus.append(consensus_pos)
                consensus_pos += 1

        # Now map the consensus-aligned sequence back.
        # consensus_aligned has the consensus with possible gaps from the pairwise alignment.
        # seq_aligned_to_consensus is the poor sequence aligned to that.
        # We need to walk through the consensus alignment to map positions.

        # Build: for each ungapped consensus position, what character does the
        # realigned sequence have?
        seq_by_consensus_pos = {}
        cons_pos = 0
        for i in range(len(consensus_aligned)):
            if consensus_aligned[i] != '-':
                seq_by_consensus_pos[cons_pos] = seq_aligned_to_consensus[i]
                cons_pos += 1
            # If consensus has gap, the seq char is an insertion relative to
            # consensus — we drop it to maintain MSA column structure

        # Build the final MSA-column-aligned sequence
        result = []
        for col in range(good_alignment.length):
            cpos = msa_to_consensus[col]
            if cpos == -1:
                # This MSA column was a gap-heavy column (not in consensus)
                result.append('-')
            elif cpos in seq_by_consensus_pos:
                result.append(seq_by_consensus_pos[cpos])
            else:
                result.append('-')

        return ''.join(result)


def compute_hit_coverage(hit: BlastHit, query_len: int, subject_len: int) -> float:
    """
    Compute the fraction of the query sequence covered by the BLAST hit.
    Returns a value between 0.0 and 1.0.
    """
    if query_len == 0:
        return 0.0
    return hit.query_len_aligned / query_len


def compute_per_sequence_identity(alignment: Alignment) -> Dict[str, float]:
    """
    Compute average pairwise percent identity for each sequence against all others.

    Returns dict mapping seq_id to its average identity (0-100).
    Useful for identifying poorly-aligned sequences.
    """
    if not alignment.is_valid() or alignment.n_seqs < 2:
        return {}

    sequences = alignment.sequences
    n_seqs = len(sequences)
    scores = {}

    for i in range(n_seqs):
        total_identity = 0.0
        n_pairs = 0

        for j in range(n_seqs):
            if i == j:
                continue

            matches = 0
            aligned_positions = 0

            for col in range(alignment.length):
                c1, c2 = sequences[i].seq[col], sequences[j].seq[col]
                if c1 != '-' and c2 != '-':
                    aligned_positions += 1
                    if c1 == c2:
                        matches += 1

            if aligned_positions > 0:
                total_identity += matches / aligned_positions
                n_pairs += 1

        scores[sequences[i].id] = (total_identity / n_pairs * 100) if n_pairs > 0 else 0.0

    return scores


def build_consensus(alignment: Alignment, gap_threshold: float = 0.5) -> str:
    """
    Build a majority-rule consensus sequence from an MSA.

    For each column:
    - If gaps exceed gap_threshold fraction, the column is skipped (not included)
    - Otherwise, the most common non-gap character is used

    Returns the ungapped consensus string.
    """
    if not alignment.is_valid() or alignment.n_seqs == 0:
        return ""

    sequences = [s.seq for s in alignment.sequences]
    n_seqs = len(sequences)
    consensus = []

    for col in range(alignment.length):
        column = [seq[col] for seq in sequences]
        gap_count = column.count('-')
        gap_fraction = gap_count / n_seqs

        if gap_fraction > gap_threshold:
            continue

        # Count non-gap characters
        char_counts = {}
        for c in column:
            if c != '-':
                char_counts[c] = char_counts.get(c, 0) + 1

        if char_counts:
            consensus.append(max(char_counts, key=char_counts.get))

    return ''.join(consensus)


def compute_msa_score(alignment: Alignment,
                      match_score: int = 1,
                      mismatch_score: int = -1,
                      gap_score: int = -1) -> float:
    """
    Compute sum-of-pairs score for an MSA.
    
    For each column, sum the pairwise scores for all sequence pairs.
    """
    if not alignment.is_valid():
        return float('-inf')
    
    total_score = 0.0
    n_seqs = alignment.n_seqs
    length = alignment.length
    
    sequences = [s.seq for s in alignment.sequences]
    
    for col in range(length):
        column = [seq[col] for seq in sequences]
        
        # Sum pairwise scores for this column
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                c1, c2 = column[i], column[j]
                
                if c1 == '-' and c2 == '-':
                    # Both gaps - no score
                    pass
                elif c1 == '-' or c2 == '-':
                    # One gap
                    total_score += gap_score
                elif c1 == c2:
                    # Match
                    total_score += match_score
                else:
                    # Mismatch
                    total_score += mismatch_score
    
    return total_score


def compute_percent_identity(alignment: Alignment) -> float:
    """
    Compute average pairwise percent identity across the MSA.
    """
    if not alignment.is_valid() or alignment.n_seqs < 2:
        return 0.0
    
    total_identity = 0.0
    n_pairs = 0
    
    sequences = [s.seq for s in alignment.sequences]
    n_seqs = len(sequences)
    
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            matches = 0
            aligned_positions = 0
            
            for col in range(alignment.length):
                c1, c2 = sequences[i][col], sequences[j][col]
                
                if c1 != '-' and c2 != '-':
                    aligned_positions += 1
                    if c1 == c2:
                        matches += 1
            
            if aligned_positions > 0:
                total_identity += matches / aligned_positions
                n_pairs += 1
    
    return (total_identity / n_pairs * 100) if n_pairs > 0 else 0.0


def compute_column_score(alignment: Alignment) -> float:
    """
    Compute total column score (penalizes non-conserved columns).
    
    Score per column:
    - All identical (no gaps): +2
    - All identical (with gaps): +1  
    - Mixed (no gaps): 0
    - Mixed (with gaps): -1
    """
    if not alignment.is_valid():
        return float('-inf')
    
    total_score = 0.0
    sequences = [s.seq for s in alignment.sequences]
    
    for col in range(alignment.length):
        column = [seq[col] for seq in sequences]
        unique_chars = set(column)
        has_gaps = '-' in unique_chars
        
        non_gap_chars = unique_chars - {'-'}
        
        if len(non_gap_chars) == 0:
            # All gaps - neutral
            pass
        elif len(non_gap_chars) == 1:
            # Conserved
            if has_gaps:
                total_score += 1
            else:
                total_score += 2
        else:
            # Variable
            if has_gaps:
                total_score -= 1
            # else: neutral
    
    return total_score
