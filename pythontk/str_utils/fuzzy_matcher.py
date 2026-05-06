# !/usr/bin/python
# coding=utf-8
from typing import Callable, List, Optional, Tuple, Dict, Union
import re


class FuzzyMatcher:
    """Fuzzy matching utilities for object names and hierarchical structures."""

    @staticmethod
    def get_base_name(name: str) -> str:
        """Remove trailing digits from name to get base name.

        Parameters:
            name: Input string that may have trailing digits

        Returns:
            String with trailing digits removed

        Examples:
            >>> FuzzyMatcher.get_base_name("object123")
            'object'
            >>> FuzzyMatcher.get_base_name("mesh_01")
            'mesh_'
        """
        return re.sub(r"\d+$", "", name)

    @staticmethod
    def find_best_match(
        target_name: str, available_names: List[str], score_threshold: float = 0.5
    ) -> Optional[Tuple[str, float]]:
        """Find best fuzzy match for target name from available candidates.

        Parameters:
            target_name: Name to find a match for
            available_names: List of candidate names to match against
            score_threshold: Minimum similarity score (0.0-1.0) to consider a match

        Returns:
            Tuple of (best_match, score) or None if no good match found

        Examples:
            >>> candidates = ["mesh_01", "mesh_02", "cube_01"]
            >>> FuzzyMatcher.find_best_match("mesh_03", candidates)
            ('mesh_01', 0.9)
        """
        # Try exact match first
        if target_name in available_names:
            return target_name, 1.0

        best_match = None
        best_score = 0.0

        for candidate in available_names:
            score = FuzzyMatcher._calculate_similarity(target_name, candidate)
            if score > best_score and score >= score_threshold:
                best_match = candidate
                best_score = score

        return (best_match, best_score) if best_match else None

    @staticmethod
    def find_all_matches(
        target_names: List[str],
        available_names: List[str],
        score_threshold: float = 0.5,
    ) -> Dict[str, Tuple[str, float]]:
        """Find fuzzy matches for multiple target names.

        Parameters:
            target_names: List of names to find matches for
            available_names: List of candidate names to match against
            score_threshold: Minimum similarity score to consider a match

        Returns:
            Dictionary mapping target_name -> (best_match, score)
        """
        matches = {}
        for target in target_names:
            match = FuzzyMatcher.find_best_match(
                target, available_names, score_threshold
            )
            if match:
                matches[target] = match
        return matches

    @staticmethod
    def find_trailing_digit_matches(
        missing_paths: List[str], extra_paths: List[str], path_separator: str = "|"
    ) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Find fuzzy matches specifically for trailing digit differences in hierarchical paths.

        This is useful for matching objects that have been renamed with different numbering
        but are otherwise identical (e.g., "mesh_01" vs "mesh_02").

        Parameters:
            missing_paths: List of paths that are missing
            extra_paths: List of paths that are extra/unexpected
            path_separator: Character used to separate hierarchy levels

        Returns:
            Tuple of:
            - fuzzy_matches: List of match dictionaries
            - paths_to_remove_from_missing: Paths that were matched and should be removed
            - paths_to_remove_from_extra: Paths that were matched and should be removed
        """
        missing_to_check = missing_paths.copy()
        extra_to_check = extra_paths.copy()

        fuzzy_matches = []
        paths_to_remove_from_missing = []
        paths_to_remove_from_extra = []

        for missing_path in missing_to_check:
            missing_node_name = missing_path.split(path_separator)[-1]
            missing_base_name = FuzzyMatcher.get_base_name(missing_node_name)

            # Skip if no trailing digits
            if missing_base_name == missing_node_name:
                continue

            for extra_path in extra_to_check:
                extra_node_name = extra_path.split(path_separator)[-1]
                extra_base_name = FuzzyMatcher.get_base_name(extra_node_name)

                if (
                    missing_base_name == extra_base_name
                    and missing_node_name != extra_node_name
                ):

                    # Check if they have the same parent
                    missing_parent = path_separator.join(
                        missing_path.split(path_separator)[:-1]
                    )
                    extra_parent = path_separator.join(
                        extra_path.split(path_separator)[:-1]
                    )

                    if missing_parent == extra_parent:
                        fuzzy_match = {
                            "current_name": extra_node_name,
                            "current_path": extra_path,
                            "target_name": missing_node_name,
                            "target_path": missing_path,
                            "parent_path": missing_parent,
                        }
                        fuzzy_matches.append(fuzzy_match)
                        paths_to_remove_from_missing.append(missing_path)
                        paths_to_remove_from_extra.append(extra_path)
                        break

        return fuzzy_matches, paths_to_remove_from_missing, paths_to_remove_from_extra

    @staticmethod
    def _calculate_similarity(
        name1: str,
        name2: str,
        use_base_name: bool = True,
        use_substring: bool = True,
        use_prefix: bool = True,
        use_ratio: bool = False,
    ) -> float:
        """Calculate similarity score between two names using multiple strategies.

        Strategies short-circuit in order: base_name → substring → prefix → ratio.
        The first one that produces a non-zero score wins.

        Parameters:
            name1: First name to compare
            name2: Second name to compare
            use_base_name: Trailing-digit base-name strategy. Disable when numbered
                variants (e.g. texture_001 vs texture_002) should NOT be considered fuzzy
                matches.
            use_substring: Substring containment strategy.
            use_prefix: Common-prefix strategy (>5 shared leading chars).
            use_ratio: difflib SequenceMatcher.ratio() — generic edit-distance similarity.
                Off by default since it's a continuous score (vs. the others' discrete
                signals) and is best used as a last-resort fallback.

        Returns:
            Similarity score from 0.0 (no similarity) to 1.0 (identical)
        """
        # Strategy 1: Base name matching (remove trailing digits)
        if use_base_name:
            base1 = FuzzyMatcher.get_base_name(name1)
            base2 = FuzzyMatcher.get_base_name(name2)

            if base1 == base2 and base1:
                if base1 != name1 or base1 != name2:
                    return 0.9
                elif name1 == name2:
                    return 1.0

        # Strategy 2: Substring matching
        if use_substring and (name1 in name2 or name2 in name1):
            return min(len(name1), len(name2)) / max(len(name1), len(name2))

        # Strategy 3: Common prefix length
        if use_prefix:
            common_prefix = 0
            for a, b in zip(name1, name2):
                if a == b:
                    common_prefix += 1
                else:
                    break

            if common_prefix > 5:  # Arbitrary threshold for meaningful prefix
                return common_prefix / max(len(name1), len(name2))

        # Strategy 4: Generic similarity ratio (difflib)
        if use_ratio:
            from difflib import SequenceMatcher

            return SequenceMatcher(None, name1, name2).ratio()

        return 0.0

    @staticmethod
    def find_unique_match(
        target: str,
        candidates: List[str],
        score_threshold: float = 0.5,
        ambiguity_delta: float = 0.05,
        use_base_name: bool = True,
        use_substring: bool = True,
        use_prefix: bool = True,
        use_ratio: bool = False,
    ) -> Tuple[Optional[str], float, str]:
        """Find a single unambiguous best match, surfacing ambiguity instead of silently picking.

        Parameters:
            target: Name to find a match for.
            candidates: Candidate names to score against.
            score_threshold: Minimum similarity required for a candidate to be considered.
            ambiguity_delta: If a runner-up scores within this delta of the best, the result
                is reported as ambiguous.
            use_base_name / use_substring / use_prefix: Toggle individual scoring strategies
                (see _calculate_similarity).

        Returns:
            Tuple of (match, score, status) where status is one of:
                - "unique":    a single clear winner (name, score populated).
                - "ambiguous": top candidate populated but a runner-up is within
                  ambiguity_delta; caller should treat as ambiguous.
                - "no_match":  no candidate met score_threshold; (None, 0.0).

        Examples:
            >>> FuzzyMatcher.find_unique_match("foo", ["demo_foo", "demo_bar"])
            ('demo_foo', 0.428..., 'unique')
        """
        if target in candidates:
            return target, 1.0, "unique"

        scored = [
            (
                c,
                FuzzyMatcher._calculate_similarity(
                    target,
                    c,
                    use_base_name=use_base_name,
                    use_substring=use_substring,
                    use_prefix=use_prefix,
                    use_ratio=use_ratio,
                ),
            )
            for c in candidates
        ]
        scored = [(c, s) for c, s in scored if s >= score_threshold]
        if not scored:
            return None, 0.0, "no_match"

        scored.sort(key=lambda cs: cs[1], reverse=True)
        best_name, best_score = scored[0]
        if len(scored) > 1 and scored[1][1] >= best_score - ambiguity_delta:
            return best_name, best_score, "ambiguous"
        return best_name, best_score, "unique"

    # --- Strategy pipeline ----------------------------------------------------

    @staticmethod
    def _builtin_strategy(
        name: str,
    ) -> Callable[[str, List[str], float, float], Tuple[Optional[str], float, str]]:
        """Resolve a built-in strategy name to a matcher callable.

        Built-ins (each runs in isolation — only the named scoring rule fires):

            "exact"     — target ∈ candidates, case-sensitive.
            "substring" — one stem contains the other.
            "prefix"    — >5 shared leading characters.
            "base_name" — trailing-digit base-name equality (e.g. mesh_01 ↔ mesh_02).
            "ratio"     — difflib SequenceMatcher.ratio() generic similarity.
            "all"       — base_name + substring + prefix (legacy find_unique_match default).
        """
        if name == "exact":

            def _exact(target, candidates, score_threshold, ambiguity_delta):
                if target in candidates:
                    dups = sum(1 for c in candidates if c == target)
                    if dups > 1:
                        return target, 1.0, "ambiguous"
                    return target, 1.0, "unique"
                return None, 0.0, "no_match"

            return _exact

        flags_by_name = {
            "substring": dict(
                use_base_name=False,
                use_substring=True,
                use_prefix=False,
                use_ratio=False,
            ),
            "prefix": dict(
                use_base_name=False,
                use_substring=False,
                use_prefix=True,
                use_ratio=False,
            ),
            "base_name": dict(
                use_base_name=True,
                use_substring=False,
                use_prefix=False,
                use_ratio=False,
            ),
            "ratio": dict(
                use_base_name=False,
                use_substring=False,
                use_prefix=False,
                use_ratio=True,
            ),
            "all": dict(
                use_base_name=True,
                use_substring=True,
                use_prefix=True,
                use_ratio=False,
            ),
        }
        if name not in flags_by_name:
            raise ValueError(
                f"Unknown built-in strategy: {name!r}. "
                f"Expected one of: exact, substring, prefix, base_name, ratio, all."
            )
        flags = flags_by_name[name]

        def _matcher(target, candidates, score_threshold, ambiguity_delta):
            return FuzzyMatcher.find_unique_match(
                target,
                candidates,
                score_threshold=score_threshold,
                ambiguity_delta=ambiguity_delta,
                **flags,
            )

        return _matcher

    @staticmethod
    def find_with_fallbacks(
        target: str,
        candidates: List[str],
        strategies: List[Union[str, Callable]],
        score_threshold: float = 0.5,
        ambiguity_delta: float = 0.05,
        stop_on_ambiguous: bool = True,
    ) -> Tuple[Optional[str], float, str, str]:
        """Try strategies in order; return on the first that produces a definitive result.

        Each strategy is run as a complete match-and-disambiguate pass. The pipeline
        advances on "no_match" only — a "unique" hit returns immediately, and
        "ambiguous" returns immediately when stop_on_ambiguous is True (the safer
        default — don't widen the search after a stricter tier already saw a tie).

        Parameters:
            target: Name to find a match for.
            candidates: Candidate names to score against.
            strategies: Ordered list. Each item is either:
                - a built-in name string (see _builtin_strategy), or
                - a callable with signature
                    (target, candidates) -> (name, score, status)
                  where status is one of "unique", "ambiguous", "no_match".
                  Custom callables close over their own configuration.
            score_threshold: Forwarded to built-in strategies.
            ambiguity_delta: Forwarded to built-in strategies.
            stop_on_ambiguous: If True (default), an "ambiguous" result halts the
                pipeline. If False, the pipeline continues to the next strategy.

        Returns:
            (match, score, status, strategy_name). strategy_name is the built-in key
            or callable's __name__ for the tier that produced the result; empty string
            when every tier returned "no_match".

        Examples:
            >>> # Texture editor: strict stem first, then containment, then ratio.
            >>> FuzzyMatcher.find_with_fallbacks(
            ...     "foo_ao",
            ...     ["demo_foo_ao", "demo_foo_diff"],
            ...     strategies=["exact", "substring", "ratio"],
            ... )
            ('demo_foo_ao', 0.46..., 'unique', 'substring')
        """
        for strat in strategies:
            if isinstance(strat, str):
                strat_name = strat
                matcher = FuzzyMatcher._builtin_strategy(strat)
                result = matcher(
                    target, candidates, score_threshold, ambiguity_delta
                )
            elif callable(strat):
                strat_name = getattr(strat, "__name__", "custom")
                result = strat(target, candidates)
            else:
                raise TypeError(
                    f"Strategy must be a name string or callable, got "
                    f"{type(strat).__name__}"
                )

            if not (isinstance(result, tuple) and len(result) == 3):
                raise ValueError(
                    f"Strategy {strat_name!r} returned malformed result: {result!r} "
                    f"(expected (name, score, status))"
                )

            name, score, status = result
            if status == "unique":
                return name, score, status, strat_name
            if status == "ambiguous" and stop_on_ambiguous:
                return name, score, status, strat_name

        return None, 0.0, "no_match", ""

    @staticmethod
    def calculate_levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein (edit) distance between two strings.

        Parameters:
            s1: First string
            s2: Second string

        Returns:
            Number of single-character edits needed to change s1 into s2
        """
        if len(s1) < len(s2):
            return FuzzyMatcher.calculate_levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def similarity_from_distance(s1: str, s2: str) -> float:
        """Calculate similarity score from Levenshtein distance.

        Parameters:
            s1: First string
            s2: Second string

        Returns:
            Similarity score from 0.0 to 1.0
        """
        distance = FuzzyMatcher.calculate_levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage and tests
    matcher = FuzzyMatcher()

    # Test basic functionality
    candidates = ["mesh_01", "mesh_02", "cube_01", "sphere_03"]

    print("=== Fuzzy Matching Examples ===")
    print(f"Candidates: {candidates}")

    test_names = ["mesh_03", "cube_02", "plane_01"]
    for name in test_names:
        match = matcher.find_best_match(name, candidates)
        if match:
            print(f"'{name}' -> '{match[0]}' (score: {match[1]:.2f})")
        else:
            print(f"'{name}' -> No good match found")

    # Test hierarchical path matching
    print("\n=== Hierarchical Path Matching ===")
    missing = ["group1|mesh_01", "group2|cube_02"]
    extra = ["group1|mesh_03", "group2|cube_05", "group3|sphere_01"]

    matches, missing_remove, extra_remove = matcher.find_trailing_digit_matches(
        missing, extra
    )

    for match in matches:
        print(f"Fuzzy match: {match['current_name']} -> {match['target_name']}")

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
# This module provides general-purpose fuzzy string matching utilities that can be used
# across different domains, not just Maya. It's particularly useful for:
#
# - Object name matching with numerical variations
# - Hierarchical path matching
# - String similarity calculations
# - Levenshtein distance calculations
#
# The module is designed to be imported and used by domain-specific tools like Maya
# hierarchy managers, but the core algorithms are generic.
# --------------------------------------------------------------------------------------------
