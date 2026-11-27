# EXP-006 bridge contract

This document records the bounded handoff from RefWorld visual R&D into the LifeOS Studio product path.

## Dependency boundary

Before EXP-006 implementation starts, resolve only:

1. the already-sealed LaMa backend-independence replication;
2. one end-to-end single-image VGGT geometry case with explicit monocular scale/pose registration and no oracle source depth.

These gates may pass or fail. Neither is allowed to expand into an open-ended prerequisite programme.

## Handoff invariant

LifeOS remains authoritative for semantic identity, project/artifact truth, state, relations, evidence, disclosure and permissions. RefWorld supplies geometry, appearance, observation provenance, hidden-space hypotheses and renderer-facing spatial assets.

A renderer must resolve spatial entities back to stable LifeOS identifiers rather than duplicate the LifeOS graph.

## Minimal proof

One owner-supplied LifeOS reference scene must support:

- reproducible hero-camera rendering;
- two neighboring views;
- three native stable semantic entities;
- leave/revisit identity continuity;
- one bounded state edit;
- edit persistence across reload/reconnect;
- unchanged unrelated semantic state;
- before/after hero-frame fidelity measurement;
- explicit observed/generated/unresolved provenance;
- mapping to at least one real LifeOS project/artifact identifier.

## Stop rule

After the two bounded dependencies above, further novel-view benchmark expansion must directly unblock reference fidelity, persistence/edit stability, or the LifeOS spatial handoff. Otherwise defer it.
