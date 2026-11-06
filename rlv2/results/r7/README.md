# Round 7: guard / persistence A/B (simulated, preset stereo)

Baseline: s120hd + RLV2_MEM_GUARD=1 + RLV2_NARROW=1, preset hd_bm320_120 + despeckle, HFOV 120, estimated attitude, RLV2_DIAG=1.
Sets: orig 63 missions x16 seeds; m3d = gen3d with under-table missions (counter, door, level, overwall, person, window) EXCLUDED, x8 seeds;
pt1 = ProcTHOR-1 preset x3; h26 = held-out ProcTHOR h26 preset x3. Run: `r7_ab.sh NAME "ENV"`; table: `ab_summary.py DIR base blank ...`.
Under-table missions are no longer part of headline scores or the gen3d aggregate (2026-10-28).

Flags: RLV2_BLANK_GUARD (mostly-blank forward sector -> cap speed 0.3 m/s + yaw dither), RLV2_HOLE_MEM (remember blank rays as
near-field suspects with 20-frame decay), RLV2_PERSIST (no progress for 3 s -> 1.5 s climb/slide/drop escape).
Result: BLANK_GUARD helps everywhere (fewest contacts); HOLE_MEM hurts (stalls, side contacts); PERSIST cuts timeouts but not contacts.
All numbers are simulated; nothing here is measured on hardware. See summary.md.
