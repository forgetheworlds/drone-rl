# Round 6 failure breakdown (simulated, box CPU)

Every failed flight of s120hd (+ memory guard + narrow route, 120°) and, for comparison, s90hd and s4
(hd_bm320_90 preset), classified by cause and mission type. All of it is simulated flight; nothing
here is measured on hardware.

## How

- `grade.cpp` `RLV2_DIAG=1` writes `CSV.diag` next to the per-flight CSV (one row per flight, fields
  listed in the grade.cpp comment). It does not change the flights: the 504-flight orig blocks and
  the ProcTHOR preset runs reproduce the round-6 numbers exactly.
- `hyperdrone_eval/failure_breakdown.py` turns the diag rows into causes (rules in its docstring).
- Sets: original 63 missions × 24 seeds (3 blocks of 8, sensor level 1, hd preset), gen3d (7 types ×
  24 missions × 8 seeds), HyperDrone ProcTHOR-1 (40 missions × 3) and held-out h26 (20 × 3).
- **ProcTHOR caveat.** Per-cause rows for ProcTHOR-1 / h26 come from the calibrated `hd_bm320_120`
  preset on the same voxelized scenes, missions and seeds (s120hd .783 / .867 vs .708 / .700 under
  real stereo). Real-stereo flights were not re-logged with per-flight diagnostics; they are split
  by outcome only, below. Headline completion and contact counts stay the real-stereo numbers.
- There is no out-of-bounds end in the grader: a flight ends on contact, on completion, or after
  40 s on one waypoint. A flight that touches anything ends there, so a completed flight never had
  a contact. "Clean" = completed and never within 0.25 m of a static surface (body radius 0.18 m).

## Real-stereo outcomes (round-6 CSVs, best camera profile)

| run | set | done | contact | timeout | contact location (in view / side / behind / floor / ceiling / diagonal) |
|---|---|---:|---:|---:|---|
| s120hd | ProcTHOR-1 | 85 | 24 | 11 | 10 / 8 / 1 / 1 / 3 / 1 |
| s90hd | ProcTHOR-1 | 81 | 27 | 12 | 10 / 7 / 0 / 3 / 2 / 5 |
| s4 | ProcTHOR-1 | 79 | 32 | 9 | 10 / 14 / 0 / 3 / 1 / 4 |
| s120hd | h26 | 42 | 14 | 4 | 6 / 2 / 1 / 3 / 1 / 1 |
| s90hd | h26 | 49 | 10 | 1 | 5 / 3 / 0 / 1 / 1 / 0 |
| s4 | h26 | 41 | 15 | 4 | 8 / 3 / 0 / 2 / 1 / 1 |

s120hd real stereo by type (done / contact / timeout): ProcTHOR-1 open 25/1/4, gap 21/9/0,
vertical 20/4/6, low_block 19/10/1; h26 open 15/0/0, gap 7/7/1, vertical 13/1/1, low_block 7/6/2.

## Findings

- **Contacts with the obstacle in view are depth holes, not control errors.** For s120hd on the
  orig set, of the rays that hit within 0.8 m at the moment of contact, stereo returned on average
  4% and 95% of those rays land on textureless wall tiles (hd preset hole rate .91 there). The local map
  held the obstacle along the velocity in only 17 of 236 in-view contacts. Only 32 of 1097 s120hd failures
  were "seen and still hit".
- **Side/behind contacts are second** (304). The memory guard only knows what stereo saw, so
  textureless walls beside the drone are also invisible to it.
- **Under-table (gen3d) is 0/192 for every student, and the true-depth teacher t3d gets 11/192.** The
  drone stalls at the table edge (timeouts end ~1 m from the goal, nearest surface 0.2-0.5 m) or
  clips the top edge (diagonal / ceiling contacts). This is a mission/controller limit, not perception.
- Moving boxes (80) are almost all fresh-crossing and person; stopped-short (69) is the arrival
  settle rule (within 1 m, never settled).


### Totals by cause

| set | flights | done | clean | contact/in-view-seen | contact/in-view-hole | contact/side-behind | contact/floor-ceiling | contact/moving-box | contact/diagonal-thin | timeout/stopped-short | timeout/stopped-at-obstacle | timeout/hover-open | timeout/no-route | timeout/wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s120hd ProcTHOR-1 preset | 120 | 94 | 80 | 1 | 7 | 6 |  |  | 6 | 6 |  |  |  |  |
| s120hd h26 preset | 60 | 52 | 48 |  | 1 | 1 | 4 |  | 1 | 1 |  |  |  |  |
| s120hd orig | 1512 | 1016 | 944 | 17 | 219 | 127 | 3 | 58 | 38 | 25 |  |  |  | 9 |
| s120hd gen3d | 1344 | 777 | 616 | 14 | 135 | 170 | 20 | 22 | 85 | 37 | 32 | 28 | 3 | 21 |
| s90hd ProcTHOR-1 preset | 120 | 83 | 71 | 1 | 5 | 17 | 2 |  | 11 | 1 |  |  |  |  |
| s90hd h26 preset | 60 | 42 | 36 |  | 9 | 4 | 2 |  |  | 3 |  |  |  |  |
| s90hd orig | 1512 | 885 | 790 | 24 | 297 | 191 | 1 | 39 | 32 | 34 |  |  |  | 9 |
| s90hd gen3d | 1344 | 743 | 600 | 11 | 200 | 124 | 19 | 21 | 99 | 27 | 35 | 52 | 3 | 10 |
| s4 ProcTHOR-1 preset | 120 | 83 | 74 | 2 | 9 | 18 | 2 |  | 5 | 1 |  |  |  |  |
| s4 h26 preset | 60 | 42 | 36 |  | 7 | 7 | 1 |  | 1 | 2 |  |  |  |  |
| s4 orig | 1512 | 865 | 795 | 26 | 288 | 207 | 6 | 45 | 41 | 31 |  |  |  | 3 |
| s4 gen3d | 1344 | 668 | 531 | 14 | 246 | 125 | 28 | 23 | 87 | 35 | 48 | 59 |  | 11 |

### s120hd ProcTHOR-1 preset

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | diagonal-thin | stopped-short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gap | 30 | 19 | 15 | 1 | 5 | 3 | 2 |  |
| low_block | 30 | 20 | 10 |  | 2 | 2 | 4 | 2 |
| open | 30 | 29 | 29 |  |  |  |  | 1 |
| vertical | 30 | 26 | 26 |  |  | 1 |  | 3 |
| **all** | 120 | 94 | 80 | 1 | 7 | 6 | 6 | 6 |

### s120hd h26 preset

| type | flights | done | clean | in-view-hole | side-behind | floor-ceiling | diagonal-thin | stopped-short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gap | 15 | 15 | 12 |  |  |  |  |  |
| low_block | 15 | 9 | 8 |  | 1 | 3 | 1 | 1 |
| open | 15 | 15 | 15 |  |  |  |  |  |
| vertical | 15 | 13 | 13 | 1 |  | 1 |  |  |
| **all** | 60 | 52 | 48 | 1 | 1 | 4 | 1 | 1 |

### s120hd orig

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | moving-box | diagonal-thin | stopped-short | wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh-crossing | 96 | 36 | 36 | 3 |  | 10 | 3 | 44 |  |  |  |
| fresh-dense | 96 | 77 | 63 |  | 7 | 10 |  |  | 2 |  |  |
| fresh-long | 72 | 64 | 57 |  |  | 5 |  |  |  | 3 |  |
| fresh-narrow | 96 | 24 | 24 |  | 48 | 21 |  |  | 3 |  |  |
| furniture | 96 | 71 | 60 |  | 4 | 20 |  |  |  | 1 |  |
| furniture_moving | 96 | 69 | 59 |  | 3 | 12 |  | 6 |  | 6 |  |
| gap_moving | 96 | 28 | 20 | 6 | 49 | 6 |  |  | 7 |  |  |
| gap_overhang | 96 | 36 | 28 |  | 41 | 13 |  |  | 6 |  |  |
| moving | 96 | 88 | 88 |  |  |  |  | 8 |  |  |  |
| narrow_gap | 96 | 32 | 24 | 3 | 42 | 16 |  |  | 3 |  |  |
| overhang | 96 | 83 | 83 |  | 7 |  |  |  | 3 | 3 |  |
| public-Complex | 192 | 182 | 182 |  |  | 3 |  |  |  | 4 | 3 |
| public-Simple | 192 | 178 | 178 |  |  |  |  |  |  | 8 | 6 |
| staggered_vertical | 96 | 48 | 42 | 5 | 18 | 11 |  |  | 14 |  |  |
| **all** | 1512 | 1016 | 944 | 17 | 219 | 127 | 3 | 58 | 38 | 25 | 9 |

### s120hd gen3d

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | moving-box | diagonal-thin | stopped-short | stopped-at-obstacle | hover-open | no-route | wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 192 | 117 | 49 | 2 | 31 | 5 |  |  | 16 | 17 |  | 2 | 2 |  |
| door | 192 | 140 | 115 | 1 | 20 | 30 |  |  |  |  |  |  | 1 |  |
| level | 192 | 190 | 163 |  |  |  | 1 |  | 1 |  |  |  |  |  |
| overwall | 192 | 164 | 159 |  | 5 | 1 |  |  | 20 | 2 |  |  |  |  |
| person | 192 | 103 | 76 | 1 | 31 | 23 | 2 | 22 | 5 | 5 |  |  |  |  |
| under | 192 | 0 | 0 | 1 | 11 | 34 | 14 |  | 40 | 13 | 32 | 26 |  | 21 |
| window | 192 | 63 | 54 | 9 | 37 | 77 | 3 |  | 3 |  |  |  |  |  |
| **all** | 1344 | 777 | 616 | 14 | 135 | 170 | 20 | 22 | 85 | 37 | 32 | 28 | 3 | 21 |

### s90hd ProcTHOR-1 preset

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | diagonal-thin | stopped-short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gap | 30 | 16 | 12 |  | 5 | 5 |  | 4 |  |
| low_block | 30 | 8 | 2 | 1 |  | 11 | 2 | 7 | 1 |
| open | 30 | 30 | 30 |  |  |  |  |  |  |
| vertical | 30 | 29 | 27 |  |  | 1 |  |  |  |
| **all** | 120 | 83 | 71 | 1 | 5 | 17 | 2 | 11 | 1 |

### s90hd h26 preset

| type | flights | done | clean | in-view-hole | side-behind | floor-ceiling | stopped-short |
|---|---:|---:|---:|---:|---:|---:|---:|
| gap | 15 | 13 | 10 |  | 2 |  |  |
| low_block | 15 | 8 | 5 | 3 | 1 | 2 | 1 |
| open | 15 | 15 | 15 |  |  |  |  |
| vertical | 15 | 6 | 6 | 6 | 1 |  | 2 |
| **all** | 60 | 42 | 36 | 9 | 4 | 2 | 3 |

### s90hd orig

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | moving-box | diagonal-thin | stopped-short | wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh-crossing | 96 | 31 | 28 | 1 | 6 | 22 |  | 35 | 1 |  |  |
| fresh-dense | 96 | 69 | 54 |  |  | 27 |  |  |  |  |  |
| fresh-long | 72 | 61 | 46 |  |  | 11 |  |  |  |  |  |
| fresh-narrow | 96 | 24 | 17 | 6 | 37 | 27 |  |  | 2 |  |  |
| furniture | 96 | 56 | 43 |  | 7 | 30 |  |  |  | 3 |  |
| furniture_moving | 96 | 60 | 41 | 3 | 10 | 18 |  | 2 |  | 3 |  |
| gap_moving | 96 | 11 | 8 | 5 | 58 | 13 |  |  | 9 |  |  |
| gap_overhang | 96 | 15 | 13 |  | 59 | 14 |  |  | 8 |  |  |
| moving | 96 | 93 | 93 |  |  |  |  | 2 |  | 1 |  |
| narrow_gap | 96 | 10 | 7 | 6 | 67 | 7 |  |  | 6 |  |  |
| overhang | 96 | 79 | 73 |  | 12 |  |  |  | 3 | 2 |  |
| public-Complex | 192 | 173 | 173 |  |  |  |  |  |  | 13 | 6 |
| public-Simple | 192 | 177 | 176 |  |  |  |  |  |  | 12 | 3 |
| staggered_vertical | 96 | 26 | 18 | 3 | 41 | 22 | 1 |  | 3 |  |  |
| **all** | 1512 | 885 | 790 | 24 | 297 | 191 | 1 | 39 | 32 | 34 | 9 |

### s90hd gen3d

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | moving-box | diagonal-thin | stopped-short | stopped-at-obstacle | hover-open | no-route | wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 192 | 82 | 29 |  | 68 | 12 |  |  | 23 | 7 |  |  |  |  |
| door | 192 | 153 | 131 |  | 18 | 21 |  |  |  |  |  |  |  |  |
| level | 192 | 187 | 158 |  |  |  |  |  | 3 | 2 |  |  |  |  |
| overwall | 192 | 134 | 128 |  | 15 | 2 | 2 |  | 38 | 1 |  |  |  |  |
| person | 192 | 101 | 78 | 1 | 43 | 12 | 4 | 21 | 5 | 5 |  |  |  |  |
| under | 192 | 0 | 0 | 4 | 12 | 33 | 4 |  | 27 | 12 | 35 | 52 | 3 | 10 |
| window | 192 | 86 | 76 | 6 | 44 | 44 | 9 |  | 3 |  |  |  |  |  |
| **all** | 1344 | 743 | 600 | 11 | 200 | 124 | 19 | 21 | 99 | 27 | 35 | 52 | 3 | 10 |

### s4 ProcTHOR-1 preset

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | diagonal-thin | stopped-short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gap | 30 | 16 | 14 |  | 5 | 6 | 1 | 2 |  |
| low_block | 30 | 10 | 5 | 2 | 4 | 9 | 1 | 3 | 1 |
| open | 30 | 29 | 29 |  |  | 1 |  |  |  |
| vertical | 30 | 28 | 26 |  |  | 2 |  |  |  |
| **all** | 120 | 83 | 74 | 2 | 9 | 18 | 2 | 5 | 1 |

### s4 h26 preset

| type | flights | done | clean | in-view-hole | side-behind | floor-ceiling | diagonal-thin | stopped-short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gap | 15 | 14 | 11 |  | 1 |  |  |  |
| low_block | 15 | 5 | 4 | 3 | 4 | 1 | 1 | 1 |
| open | 15 | 15 | 15 |  |  |  |  |  |
| vertical | 15 | 8 | 6 | 4 | 2 |  |  | 1 |
| **all** | 60 | 42 | 36 | 7 | 7 | 1 | 1 | 2 |

### s4 orig

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | moving-box | diagonal-thin | stopped-short | wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh-crossing | 96 | 33 | 30 |  | 5 | 15 |  | 42 |  | 1 |  |
| fresh-dense | 96 | 54 | 48 | 3 |  | 39 |  |  |  |  |  |
| fresh-long | 72 | 51 | 42 |  | 11 | 7 |  |  |  | 3 |  |
| fresh-narrow | 96 | 21 | 12 | 3 | 43 | 27 |  |  | 2 |  |  |
| furniture | 96 | 70 | 52 |  | 2 | 24 |  |  |  |  |  |
| furniture_moving | 96 | 59 | 46 |  | 12 | 21 |  | 1 |  | 3 |  |
| gap_moving | 96 | 13 | 10 | 3 | 58 | 14 | 3 | 2 | 3 |  |  |
| gap_overhang | 96 | 15 | 12 | 11 | 49 | 14 |  |  | 7 |  |  |
| moving | 96 | 93 | 93 |  |  |  |  |  |  | 3 |  |
| narrow_gap | 96 | 10 | 10 | 3 | 50 | 19 |  |  | 14 |  |  |
| overhang | 96 | 78 | 75 |  | 4 | 6 |  |  | 8 |  |  |
| public-Complex | 192 | 178 | 178 |  |  |  | 3 |  |  | 8 | 3 |
| public-Simple | 192 | 176 | 176 |  |  |  |  |  | 3 | 13 |  |
| staggered_vertical | 96 | 14 | 11 | 3 | 54 | 21 |  |  | 4 |  |  |
| **all** | 1512 | 865 | 795 | 26 | 288 | 207 | 6 | 45 | 41 | 31 | 3 |

### s4 gen3d

| type | flights | done | clean | in-view-seen | in-view-hole | side-behind | floor-ceiling | moving-box | diagonal-thin | stopped-short | stopped-at-obstacle | hover-open | wandering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 192 | 67 | 19 | 1 | 68 | 18 | 1 |  | 23 | 14 |  |  |  |
| door | 192 | 112 | 93 | 1 | 54 | 24 |  |  |  | 1 |  |  |  |
| level | 192 | 179 | 154 |  |  |  | 3 |  | 7 | 3 |  |  |  |
| overwall | 192 | 140 | 126 |  | 16 | 3 |  |  | 29 | 4 |  |  |  |
| person | 192 | 86 | 67 | 4 | 42 | 19 | 5 | 23 | 5 | 7 |  |  | 1 |
| under | 192 | 0 | 0 | 5 | 9 | 23 | 11 |  | 21 | 6 | 48 | 59 | 10 |
| window | 192 | 84 | 72 | 3 | 57 | 38 | 8 |  | 2 |  |  |  |  |
| **all** | 1344 | 668 | 531 | 14 | 246 | 125 | 28 | 23 | 87 | 35 | 48 | 59 | 11 |
