"""Phase 5 — experiment harness, runner support, and analysis (Phase 5).

Contains the episode scenario resolution, tau-star calibration bridge,
ablation row resolution, C3 compose-override mechanism, in-process
episode harness (faithful to the frozen Phase 2/3/4 components), raw
record building/validation, and the experiment runner + analyzer
implementations.

Design constraint (research integrity): these modules import and reuse
the frozen components (client.buffer, tool_world.ledger,
reintegration service + baselines, pilot.key_schemes, pilot.wilson,
experiments.outage_model); they never modify frozen methodology.
"""