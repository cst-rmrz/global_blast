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


def needleman_wunsch(seq1: str, seq2: str,
                     match: int = 2, mismatch: int = -1,
                     gap: int = -2) -> Tuple[str, str]:
    """
    Global pairwise alignment via Needleman-Wunsch dynamic programming.

    Used when BLAST coverage is too low to build a reliable alignment —
    typically for highly divergent sequences or sequences with large
    internal insertions. Distributes gaps throughout rather than
    end-padding like terminal extension does.

    Returns two aligned strings of equal length (with '-' for gaps).
    """
    n, m = len(seq1), len(seq2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap

    for i in range(1, n + 1):
        row_prev = dp[i - 1]
        row_curr = dp[i]
        s1i = seq1[i - 1].upper()
        for j in range(1, m + 1):
            diag = row_prev[j - 1] + (match if s1i == seq2[j - 1].upper() else mismatch)
            up   = row_prev[j] + gap
            left = row_curr[j - 1] + gap
            row_curr[j] = diag if diag >= up and diag >= left else (up if up >= left else left)

    aligned1: List[str] = []
    aligned2: List[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s1i = seq1[i - 1].upper()
            diag_score = dp[i - 1][j - 1] + (match if s1i == seq2[j - 1].upper() else mismatch)
            if dp[i][j] == diag_score:
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + gap):
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    return ''.join(aligned1), ''.join(aligned2)


def nw_aligned_pair(seq1: Sequence, seq2: Sequence) -> AlignedPair:
    """Wrap needleman_wunsch() as an AlignedPair (same interface as extend_pairwise_alignment)."""
    a1, a2 = needleman_wunsch(seq1.seq, seq2.seq)
    return AlignedPair(
        seq1_id=seq1.id, seq2_id=seq2.id,
        seq1_aligned=a1, seq2_aligned=a2,
        seq1_original=seq1.seq, seq2_original=seq2.seq
    )


def extend_pairwise_alignment(hit: BlastHit,
                              query_seq: str,
                              subject_seq: str,
                              nw_threshold: float = 0.0) -> AlignedPair:
    """
    Extend a local BLAST alignment to cover full sequence lengths.

    Strategy:
    1. Handle unaligned N-terminal regions
    2. Keep the BLAST-aligned middle region
    3. Handle unaligned C-terminal regions

    When nw_threshold > 0 and BLAST coverage of the query is below that
    fraction, terminal overhangs are aligned with Needleman-Wunsch instead
    of raw gap-padding. This distributes gaps within the tails rather than
    stacking them at the ends, which improves alignment quality for
    divergent sequences.
    """
    query_aln = hit.query_seq
    subject_aln = hit.subject_seq

    query_n_term = query_seq[:hit.query_start - 1]
    subject_n_term = subject_seq[:hit.subject_start - 1]
    query_c_term = query_seq[hit.query_end:]
    subject_c_term = subject_seq[hit.subject_end:]

    # Decide whether to use NW for terminal regions
    use_nw = False
    if nw_threshold > 0 and len(query_seq) > 0:
        coverage = hit.query_len_aligned / len(query_seq)
        use_nw = coverage < nw_threshold

    if use_nw:
        if query_n_term and subject_n_term:
            n_term_query, n_term_subject = needleman_wunsch(query_n_term, subject_n_term)
        else:
            n_term_query, n_term_subject = _extend_terminal(
                query_n_term, subject_n_term, is_n_terminal=True)

        if query_c_term and subject_c_term:
            c_term_query, c_term_subject = needleman_wunsch(query_c_term, subject_c_term)
        else:
            c_term_query, c_term_subject = _extend_terminal(
                query_c_term, subject_c_term, is_n_terminal=False)
    else:
        n_term_query, n_term_subject = _extend_terminal(
            query_n_term, subject_n_term, is_n_terminal=True)
        c_term_query, c_term_subject = _extend_terminal(
            query_c_term, subject_c_term, is_n_terminal=False)

    full_query = n_term_query + query_aln + c_term_query
    full_subject = n_term_subject + subject_aln + c_term_subject

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
                  nw_terminals: float = 0.0,
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
                
                pair = extend_pairwise_alignment(hit, center_seq.seq, other_seq.seq,
                                                nw_threshold=nw_terminals)
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

    def iterative_refine(self, alignment: Alignment,
                         gap_open: int, gap_extend: int,
                         word_size: int, evalue: float,
                         identity_threshold: float = 70.0,
                         max_iterations: int = 5,
                         coverage_threshold: float = 0.5,
                         max_hsps: int = 1,
                         nw_subgroup: bool = True,
                         nw_terminals: float = 0.0,
                         verbose: bool = False) -> Alignment:
        """
        Iterative center-star refinement for poorly-aligned sequences.

        Each iteration:
          1. Find sequences below identity_threshold (within-subgroup check)
          2. Build a new center-star MSA from only those sequences using NW
             pairwise alignments (when nw_subgroup=True) or BLAST
          3. Merge the sub-MSA back into the main alignment via NW bridge
          4. Repeat with sequences still below threshold within their sub-group

        nw_subgroup: use Needleman-Wunsch for sub-group pairwise alignment
                     instead of BLAST (option 3)
        nw_terminals: coverage threshold below which terminal overhangs in
                      the main build_msa step are NW-aligned instead of
                      gap-padded (option 2); 0 disables
        """
        scores = compute_per_sequence_identity(alignment)
        current_poor = {sid for sid, s in scores.items() if s < identity_threshold}

        if verbose:
            print(f"  Iterative centerstar: {len(current_poor)} sequences "
                  f"below {identity_threshold:.0f}% identity")

        for iteration in range(max_iterations):
            if len(current_poor) < 2:
                if verbose:
                    print(f"  Iteration {iteration + 1}: group too small, stopping")
                break

            if verbose:
                print(f"  Iteration {iteration + 1}: sub-aligning "
                      f"{len(current_poor)} sequences"
                      + (" (NW)" if nw_subgroup else " (BLAST)") + "...")

            sub_seqs = [self.sequences[sid] for sid in current_poor
                        if sid in self.sequences]

            if nw_subgroup:
                # Build all pairwise NW alignments directly — no BLAST needed
                sub_hits = {}
                for i, s1 in enumerate(sub_seqs):
                    for s2 in sub_seqs[i + 1:]:
                        pair = nw_aligned_pair(s1, s2)
                        # Store as a synthetic BlastHit covering the full sequences
                        hit_fwd = BlastHit(
                            query_id=s1.id, subject_id=s2.id,
                            query_start=1, query_end=len(s1.seq),
                            subject_start=1, subject_end=len(s2.seq),
                            query_seq=pair.seq1_aligned.replace('-', ''),
                            subject_seq=pair.seq2_aligned.replace('-', ''),
                            evalue=0.0, bitscore=1.0,
                            identity=sum(a == b for a, b in zip(
                                pair.seq1_aligned, pair.seq2_aligned)
                                if a != '-' and b != '-') /
                                max(1, sum(a != '-' or b != '-' for a, b in zip(
                                    pair.seq1_aligned, pair.seq2_aligned))) * 100
                        )
                        sub_hits[(s1.id, s2.id)] = hit_fwd
                        sub_hits[(s2.id, s1.id)] = BlastHit(
                            query_id=s2.id, subject_id=s1.id,
                            query_start=1, query_end=len(s2.seq),
                            subject_start=1, subject_end=len(s1.seq),
                            query_seq=pair.seq2_aligned.replace('-', ''),
                            subject_seq=pair.seq1_aligned.replace('-', ''),
                            evalue=0.0, bitscore=1.0,
                            identity=hit_fwd.identity
                        )
            else:
                with BlastRunner(self.seq_type) as runner:
                    sub_hits = runner.run_all_pairwise(
                        sub_seqs,
                        gap_open=gap_open,
                        gap_extend=gap_extend,
                        word_size=word_size,
                        evalue=evalue,
                        verbose=False,
                        coverage_threshold=coverage_threshold,
                        max_hsps=max_hsps
                    )

            # Build sub-MSA with auto-selected center
            sub_aligner = CenterStarAligner(sub_seqs, self.seq_type)
            sub_msa = sub_aligner.build_msa(sub_hits, nw_terminals=nw_terminals)

            sub_scores = compute_per_sequence_identity(sub_msa)
            still_poor = {sid for sid, s in sub_scores.items()
                          if s < identity_threshold}

            auto_improved = current_poor - still_poor

            if verbose:
                print(f"    Auto-center: {len(auto_improved)} improved, "
                      f"{len(still_poor)} still below threshold")

            if not auto_improved:
                # Auto-center stalled — try each candidate as center
                if verbose:
                    print(f"    Stalled. Trying exhaustive center search "
                          f"({len(current_poor)} candidates)...")

                best_sub_msa = sub_msa
                best_still_poor = still_poor
                found_improvement = False

                for candidate_id in sorted(current_poor):
                    cand_aligner = CenterStarAligner(sub_seqs, self.seq_type)
                    cand_msa = cand_aligner.build_msa(sub_hits, center_id=candidate_id,
                                                        nw_terminals=nw_terminals)
                    cand_scores = compute_per_sequence_identity(cand_msa)
                    cand_still_poor = {sid for sid, s in cand_scores.items()
                                       if s < identity_threshold}
                    cand_improved = current_poor - cand_still_poor

                    if verbose:
                        print(f"      Center {candidate_id}: "
                              f"{len(cand_improved)} improved")

                    if cand_improved:
                        best_sub_msa = cand_msa
                        best_still_poor = cand_still_poor
                        found_improvement = True
                        break

                alignment = self._merge_sub_msa(
                    alignment, best_sub_msa, current_poor, verbose,
                    max_hsps=max_hsps)
                current_poor = best_still_poor

                if not found_improvement:
                    if verbose:
                        print(f"    Exhaustive search exhausted, stopping")
                    break
            else:
                alignment = self._merge_sub_msa(
                    alignment, sub_msa, current_poor, verbose,
                    max_hsps=max_hsps)
                current_poor = still_poor

            if not current_poor:
                if verbose:
                    print(f"  All sequences resolved after {iteration + 1} iteration(s)")
                break

        return alignment

    def _merge_sub_msa(self, main_alignment: Alignment, sub_msa: Alignment,
                        poor_ids: set, verbose: bool = False,
                        max_hsps: int = 3) -> Alignment:
        """
        Merge a sub-MSA back into the main alignment using a bridge alignment.

          1. Build consensus from the well-aligned (non-poor) sequences in main
          2. Build consensus from the sub-MSA
          3. Needleman-Wunsch global alignment of sub-consensus vs main-consensus
          4. For each poor sequence: map sub-MSA columns → sub-consensus positions
             → main-consensus positions → main-MSA columns (double mapping)
          5. Replace poor sequences in main alignment with remapped versions
        """
        good_seqs = [s for s in main_alignment.sequences if s.id not in poor_ids]
        if len(good_seqs) < 2:
            if verbose:
                print("    Too few good sequences for bridge, skipping merge")
            return main_alignment

        good_alignment = Alignment(sequences=good_seqs, seq_type=main_alignment.seq_type)
        main_consensus = build_consensus(good_alignment, gap_threshold=0.5)
        sub_consensus = build_consensus(sub_msa, gap_threshold=0.5)

        if not main_consensus or not sub_consensus:
            if verbose:
                print("    Could not build consensus for bridge, skipping merge")
            return main_alignment

        # Bridge: global NW alignment of sub-consensus vs main-consensus
        if verbose:
            print(f"    Bridge NW (sub-consensus {len(sub_consensus)}bp "
                  f"vs main-consensus {len(main_consensus)}bp)")
        sub_cons_aln, main_cons_aln = needleman_wunsch(sub_consensus, main_consensus)

        # Wrap as an AlignedPair so downstream mapping code is unchanged
        bridge_pair = AlignedPair(
            seq1_id='sub_consensus', seq2_id='main_consensus',
            seq1_aligned=sub_cons_aln, seq2_aligned=main_cons_aln,
            seq1_original=sub_consensus, seq2_original=main_consensus
        )

        # Precompute: for each poor sequence, its characters at sub-consensus positions
        sub_msa_seqs = {s.id: s.seq for s in sub_msa.sequences}

        refined_seqs = list(good_seqs)

        for sid in poor_ids:
            if sid not in sub_msa_seqs:
                orig = next((s for s in main_alignment.sequences if s.id == sid), None)
                if orig:
                    refined_seqs.append(orig)
                continue

            seq_chars = self._seq_chars_at_sub_consensus(sub_msa_seqs[sid], sub_msa)

            # Double-map: sub-MSA → sub-consensus positions → bridge space → main-MSA columns
            seq_in_bridge = self._map_through_bridge(
                seq_chars, bridge_pair.seq1_aligned, bridge_pair.seq2_aligned
            )
            aligned_seq = self._map_to_msa_columns(
                seq_in_bridge, bridge_pair.seq2_aligned, good_alignment
            )

            orig_seq = self.sequences[sid]
            refined_seqs.append(Sequence(
                id=sid,
                description=orig_seq.description,
                seq=aligned_seq
            ))

        max_len = max(len(s.seq) for s in refined_seqs)
        for s in refined_seqs:
            if len(s.seq) < max_len:
                s.seq = s.seq + '-' * (max_len - len(s.seq))

        return Alignment(
            sequences=refined_seqs,
            seq_type=main_alignment.seq_type,
            parameters=main_alignment.parameters,
            score=main_alignment.score
        )

    def _seq_chars_at_sub_consensus(self, seq_in_sub_msa: str,
                                     sub_msa: Alignment) -> Dict[int, str]:
        """
        Derive the character a sequence has at each sub-consensus position.

        Walks sub-MSA columns; non-gap-heavy columns contribute to the
        sub-consensus (same logic as build_consensus). Returns a dict mapping
        sub-consensus position (0-based) → the sequence's character there.
        """
        sequences = [s.seq for s in sub_msa.sequences]
        n_seqs = len(sequences)
        result = {}
        sub_cons_pos = 0

        for col in range(sub_msa.length):
            column = [seq[col] for seq in sequences]
            if column.count('-') / n_seqs > 0.5:
                continue  # gap-heavy: skipped in sub-consensus
            result[sub_cons_pos] = seq_in_sub_msa[col]
            sub_cons_pos += 1

        return result

    def _map_through_bridge(self, seq_chars: Dict[int, str],
                             sub_cons_aligned: str,
                             main_cons_aligned: str) -> str:
        """
        Map a sequence (by its sub-consensus-position characters) into the
        bridge pairwise alignment coordinate space (aligned to main-consensus).

        sub_cons_aligned and main_cons_aligned are the two sides of the extended
        bridge alignment (same length). Returns a string of the same length
        representing the sequence aligned to main-consensus.
        """
        result = []
        sub_cons_pos = 0

        for i in range(len(sub_cons_aligned)):
            if sub_cons_aligned[i] != '-':
                result.append(seq_chars.get(sub_cons_pos, '-'))
                sub_cons_pos += 1
            else:
                # Main-consensus has an insertion here relative to sub-consensus
                result.append('-')

        return ''.join(result)

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
