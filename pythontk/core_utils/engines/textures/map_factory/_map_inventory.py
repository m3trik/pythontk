# !/usr/bin/python
# coding=utf-8
"""A texture set's inventory pruned before processing: duplicate normal maps
resolved, and maps a packed map already carries dropped (the bodies behind
the :class:`MapFactory` facade).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set


class _MapInventoryInternal:
    """Bodies of :class:`MapFactory`'s redundant-map pruning.

    The public signatures and docstrings stay on :class:`MapFactory`; each
    delegates here. A ``MapFactory`` base, so ``cls`` is the factory (or a
    subclass) and its registries and helpers resolve through the MRO.
    """

    @classmethod
    def resolve_normal_maps(
        cls,
        sorted_maps: Dict[str, Any],
        target_format: Optional[str] = None,
        convert: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """Body of :meth:`MapFactory.resolve_normal_maps`."""
        report: Dict[str, Dict[str, str]] = {"dropped": {}, "converted": {}}
        registry = cls._map_registry

        present = [t for t in registry.NORMAL_TYPES if t in sorted_maps]
        winner_type = registry.select_normal_type(sorted_maps)
        if winner_type is None:
            return report

        for map_type in present:
            if map_type == winner_type:
                continue
            cls.logger.info(
                f"Skipping {map_type} map (superseded by the {winner_type} normal map)",
                extra={"preset": "highlight"},
            )
            del sorted_maps[map_type]
            report["dropped"][map_type] = f"superseded by the {winner_type} normal map"

        if not target_format or not convert:
            return report

        # Convention name -> canonical map type, derived from the registry so a
        # newly registered tagged type is understood without editing this, and
        # so the caller's spelling is normalized. `convert_normal_map_format`
        # takes its format case-insensitively, so a caller passing "opengl" is
        # reasonable -- and would otherwise build the map type "Normal_opengl",
        # putting a type nothing in the taxonomy recognises into the inventory.
        conventions = {
            t[len("Normal_") :].lower(): t
            for t in registry.NORMAL_TYPES
            if t.startswith("Normal_")
        }
        target_type = conventions.get(str(target_format).strip().lower())
        if target_type is None:
            cls.logger.warning(
                f"Unknown normal map convention {target_format!r} "
                f"(expected one of {sorted(conventions)}); leaving the map as-is."
            )
            return report

        # Only an explicitly tagged opposite is convertible; the generic map's
        # convention is unknown and a match needs no work.
        if winner_type == target_type or winner_type not in conventions.values():
            return report

        value = sorted_maps[winner_type]
        is_list = isinstance(value, (list, tuple))
        source = (value[0] if value else None) if is_list else value
        if not source:
            return report  # nothing to convert (empty entry)
        try:
            converted = cls.convert_normal_map_format(
                source, target_format=target_format.lower()
            )
        except Exception as error:
            cls.logger.warning(
                f"Could not convert {winner_type} to {target_type} ({error}); "
                "keeping the source map."
            )
            return report

        if not converted:
            return report

        del sorted_maps[winner_type]
        sorted_maps[target_type] = [converted] if is_list else converted
        # The source type leaves the inventory, so it belongs in `dropped` too:
        # a caller mapping the report back onto its own list of real file paths
        # would otherwise keep the original file alongside the converted one and
        # wire the normal slot twice — the exact double-wire this method exists
        # to prevent. Every type that leaves the inventory is reported.
        report["dropped"][winner_type] = f"converted to {target_type}"
        report["converted"][target_type] = converted
        return report

    @classmethod
    def filter_redundant_maps(
        cls,
        sorted_maps: Dict[str, Any],
        config: Dict[str, Any] = None,
        extract_missing: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """Body of :meth:`MapFactory.filter_redundant_maps`."""
        report: Dict[str, Dict[str, str]] = {"dropped": {}, "extracted": {}}
        precedence_rules = cls.get_precedence_rules()
        registry = cls._map_registry

        def drop(map_type: str, reason: str) -> None:
            cls._drop_map(sorted_maps, report, map_type, reason)

        # Rival PACKED maps first: `replaces` cannot express that conflict, so
        # leaving it to the loop below lets the requested packing retire the
        # loose components and the rival then read as a sole source.
        cls._resolve_packed_conflicts(sorted_maps, config, extract_missing, report)

        for dominant, declared_redundants in precedence_rules.items():
            if not (dominant in sorted_maps and sorted_maps[dominant]):
                continue

            # LOOSE components only. A `replaces` entry naming another PACKING
            # (MSAO lists Metallic_Smoothness) would let this pass retire a
            # rival on name alone — no ranking, no coverage check — and so
            # overturn the packed-vs-packed pass above, which may have kept
            # both deliberately because dropping one would lose a channel.
            redundants = [r for r in declared_redundants if cls._is_loose(r)]

            # Does the target workflow actually want this packed map as output?
            # Default True keeps legacy "packed wins" behavior when no config.
            packed_requested = True
            map_def = registry.get(dominant)
            if config is not None:
                key = map_def.config_key if map_def else None
                if key:
                    packed_requested = (
                        bool(config.get(key))
                        or registry.resolve_missing_map_rule(config)
                        != registry.MISSING_SKIP
                    )

            if packed_requested:
                # Packed map supersedes its loose components.
                for redundant in redundants:
                    if redundant in sorted_maps:
                        drop(redundant, f"superseded by {dominant}")
                continue

            # Unpacked workflow. Judge coverage per declared channel: covered
            # when the carried type — or a loose type a registered conversion
            # derives it from — survives the drop. With no loose components at
            # all nothing is covered, so this extracts the whole packing, which
            # is the point: the preset decides the shape, not the input set.
            present = {
                t
                for t, v in sorted_maps.items()
                if v and t != dominant and cls._is_loose(t)
            }

            carried = map_def.carried_types() if map_def else []
            if not (
                carried or any(r in sorted_maps and sorted_maps[r] for r in redundants)
            ):
                # A packing that carries nothing checkable, with no loose
                # component standing by to take the slots over: coverage cannot
                # be judged and there is nothing to extract, so the drop below
                # would be a pure loss. ``MapType.__post_init__`` already
                # refuses a packing with no ``channels`` at all, which leaves
                # one shape — every channel marked OPTIONAL, so
                # ``carried_types()`` skips them all. Rare, and only reachable
                # through a caller-registered type, but the failure is a map
                # silently vanishing, and keeping it is this function's standing
                # answer to "cannot be shown redundant". A packing that carries
                # real channels needs no guard: they are extracted first.
                continue

            uncovered = [t for t in carried if not cls._channel_covered(t, present)]

            if uncovered:
                extracted = None
                if extract_missing:
                    extracted = cls.extract_channels(
                        dominant,
                        cls._first_path(sorted_maps[dominant]),
                        uncovered,
                        config,
                    )
                if extracted is None:
                    # Can't recover the uncovered channels — keeping the packed
                    # map is the only lossless direction; its loose components
                    # retire so the slots still have one source each.
                    reason = (
                        f"superseded by {dominant} (its "
                        f"{', '.join(uncovered)} channel has no loose source; "
                        "extraction unavailable)"
                    )
                    for redundant in redundants:
                        if redundant in sorted_maps:
                            drop(redundant, reason)
                    continue

                cls._absorb_extracted(sorted_maps, report, dominant, extracted)

            drop(dominant, "superseded by separate maps")

        return report

    @staticmethod
    def _first_path(value) -> Optional[str]:
        """The single path behind an inventory value (path or list of paths)."""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return value if isinstance(value, str) else None

    @classmethod
    def _drop_map(
        cls,
        sorted_maps: Dict[str, Any],
        report: Dict[str, Dict[str, str]],
        map_type: str,
        reason: str,
    ) -> None:
        """Remove ``map_type`` from the inventory and record why."""
        cls.logger.info(
            f"Skipping {map_type} map ({reason})",
            extra={"preset": "highlight"},
        )
        del sorted_maps[map_type]
        report["dropped"][map_type] = reason

    @classmethod
    def _absorb_extracted(
        cls,
        sorted_maps: Dict[str, Any],
        report: Dict[str, Dict[str, str]],
        source_type: str,
        extracted: Dict[str, str],
    ) -> None:
        """Add extracted loose maps to the inventory, keeping its value shape."""
        as_list = isinstance(sorted_maps[source_type], (list, tuple))
        for map_type, path in extracted.items():
            sorted_maps[map_type] = [path] if as_list else path
            report["extracted"][map_type] = path
            cls.logger.info(
                f"Extracted {map_type} from {source_type} ({path})",
                extra={"preset": "highlight"},
            )

    @classmethod
    def _is_loose(cls, map_type: str) -> bool:
        """Is ``map_type`` a separate map rather than a packing?

        Unknown types count as loose: a type the registry does not define
        cannot be asserted to pack anything.
        """
        map_def = cls._map_registry.get(map_type)
        return not (map_def and map_def.is_packed)

    @classmethod
    def _channel_covered(cls, carried_type: str, present: Set[str]) -> bool:
        """Does anything in ``present`` source ``carried_type``?

        Covered when the type is present outright, or when a single-source
        conversion derives it from something that is. That second arm is what
        makes the judgement work across packings as well as loose maps: the
        registry declares ``Roughness <- Smoothness`` *and*
        ``Roughness <- ORM``, so a surviving ORM demonstrably covers an MSAO's
        metallic / AO / smoothness channels. Shared by both redundancy passes
        so that "covered" means one thing.
        """
        if carried_type in present:
            return True
        for conv in cls._conversion_registry.get_conversions_for(carried_type):
            if len(conv.source_types) == 1 and conv.source_types[0] in present:
                return True
        return False

    @classmethod
    def _resolve_packed_conflicts(
        cls,
        sorted_maps: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        extract_missing: bool,
        report: Dict[str, Dict[str, str]],
    ) -> None:
        """Reduce rival PACKED maps to the one this workflow wants.

        The packed-vs-loose pass cannot do this. Precedence is keyed off
        :attr:`MapType.replaces`, which lists the LOOSE maps a packing absorbs
        — no packing lists another — so two packings never meet there. Worse,
        the loose pass actively hides the conflict: the requested packing
        retires the loose Metallic/Roughness/AO first, after which the rival
        finds no loose components left and takes the "sole source of its
        channels" branch. Measured live on a glTF 2.0 conversion, which
        connected an ORM *and* an HDRP MSAO to the same three slots.

        Rivalry is judged by channel coverage, not by name: a packing is a
        rival only when a more-preferred one covers at least one channel it
        carries, so ``Albedo_Transparency`` (base colour + opacity) is never
        weighed against an ORM. Preference is
        :meth:`MapRegistry.packed_precedence` — a total order, so the survivor
        does not depend on which rival was judged first.

        Lossless, like the loose pass: channels nothing else supplies are
        extracted from the loser before it is dropped, and if extraction is
        unavailable the loser is KEPT (with a warning) rather than taking its
        only copy of a channel with it. "Nothing else" counts surviving LOOSE
        maps as well as the winners — otherwise a channel the caller already
        listed loose is extracted anyway, and their entry is replaced by
        derived data.

        Modifies ``sorted_maps`` and ``report`` in place.
        """
        registry = cls._map_registry
        order = [
            t
            for t in registry.packed_precedence(config)
            if t in sorted_maps and sorted_maps[t]
        ]
        if len(order) < 2:
            return

        # Least-preferred first: a loser is judged against the survivors that
        # outrank it, so a three-way pile-up collapses in one pass. Descending
        # is what makes `order[:rank]` safe to use unfiltered — only indices
        # ABOVE the current one have been dropped, so every more-preferred
        # entry is still in `sorted_maps`.
        for rank in range(len(order) - 1, 0, -1):
            loser = order[rank]
            winners = order[:rank]

            map_def = registry.get(loser)
            carried = map_def.carried_types() if map_def else []

            # Rivalry is decided against the WINNERS alone. A channel some
            # loose map happens to cover says nothing about whether two
            # PACKINGS collide — an ORM beside a loose Metallic is the loose
            # pass's business, and crediting that here would make the ORM look
            # like a rival of whatever packing outranked it.
            rivals = set(winners)
            uncovered = [t for t in carried if not cls._channel_covered(t, rivals)]
            if len(uncovered) == len(carried):
                continue  # shares no channel with any winner — not a rival

            # For what must be EXTRACTED, a surviving loose map counts too:
            # re-extracting a channel one already supplies swaps the caller's
            # own entry for derived data (measured: an `asset_Mixed_AO.png`
            # replaced by an extracted `asset_Ambient_Occlusion.png` — the
            # canonical-name guard in `extract_channels` cannot
            # catch it, since the caller's file need not use that name).
            # Excluded are any the winners will retire below: `replaces` may
            # name a type its packing does not carry (MSAO lists Specular),
            # which would leave that channel with no source at all.
            retired = {r for w in winners for r in registry.get(w).replaces}
            loose = {
                t
                for t, v in sorted_maps.items()
                if v and t != loser and t not in retired and cls._is_loose(t)
            }
            uncovered = [t for t in uncovered if not cls._channel_covered(t, loose)]

            if uncovered:
                extracted = (
                    cls.extract_channels(
                        loser,
                        cls._first_path(sorted_maps[loser]),
                        uncovered,
                        config,
                    )
                    if extract_missing
                    else None
                )
                if extracted is None:
                    # Keeping both double-wires the shared slots, but dropping
                    # the loser would lose a channel outright. Say so instead
                    # of picking silently.
                    cls.logger.warning(
                        f"{loser} loses to {winners[0]} as a rival packing, but "
                        f"its {', '.join(uncovered)} channel has no other "
                        "source and extraction is unavailable — keeping both. "
                        "They will drive the same material slots."
                    )
                    continue
                cls._absorb_extracted(sorted_maps, report, loser, extracted)

            cls._drop_map(
                sorted_maps,
                report,
                loser,
                f"superseded by {winners[0]} (rival packing for the same channels)",
            )
