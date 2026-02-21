# Optimized Solver for the 45-Hole Cross Board

The optimized solver (press `o`) can solve the 9x9 cross-shaped board with 45 holes
in about 70 seconds. This document explains how it works, with direct references to
the source code in `peg_solitaire.py` and concrete examples of what each optimization
does at runtime.

## The problem

The 45-hole cross board has 44 initial pegs (all holes filled except center) and
requires 43 moves to reach the goal of a single peg at the center. The search tree
is enormous: each position can have 4-10 valid moves, leading to a branching factor
that makes naive backtracking infeasible.

Consider: with an average of 7 valid moves per position and 43 levels, the raw search
tree has on the order of 7^43 ~ 10^36 nodes. Even exploring a tiny fraction of this
tree requires every operation to be as fast as possible.

## Key optimizations

### 1. Bitboard representation

**The idea**: instead of storing piece positions as a Python `set` of `(x, y)` tuples,
encode the entire board state as a single 45-bit integer. Each of the 45 playable
positions is assigned a bit index (0-44).

**Position mapping** (lines 20-28):
```python
PLAYABLE_POSITIONS = []
POS_TO_BIT = {}
for _y in range(N):
    for _x in range(N):
        if standard_board[_y][_x] != -1:
            POS_TO_BIT[(_x, _y)] = len(PLAYABLE_POSITIONS)
            PLAYABLE_POSITIONS.append((_x, _y))
```

This scans the board row by row and assigns consecutive bit indices to each playable
cell. For instance, the top arm positions `(3,0)`, `(4,0)`, `(5,0)` get indices 0, 1, 2,
and the center `(4,4)` gets index 22.

**Move precomputation** (lines 31-53): each directed jump is stored as a tuple of
bitmasks:
```python
MOVES.append((_fb, _jb, _tb, _fm, _jm, _tm, _fm | _jm | _tm))
```

Where `_fm = 1 << _fb` is the single-bit mask for the source position, and
`_fm | _jm | _tm` is the combined XOR mask.

**How a move works at runtime** (line 588):
```python
new_state = cur_state ^ xm
```

A single XOR simultaneously removes the source peg, removes the jumped peg, and
places the destination peg, because all three bits flip. Undoing the move is the
exact same XOR.

**Example**: suppose a peg at bit 0 jumps over bit 1 to land on bit 2.
The XOR mask is `0b111` (bits 0, 1, and 2).

```
Before:  ...1 1 0    (pegs at 0 and 1, empty at 2)
XOR:     ...1 1 1
After:   ...0 0 1    (empty at 0 and 1, peg at 2)
```

Compare this to the old solver's `_play` method (lines 242-250), which requires
three separate `_set()` calls, each doing a `set.discard()` and `set.add()`:
```python
self._set(pos1[0],pos1[1],0)          # remove source
self._set(pos2[0],pos2[1],1)          # place destination
self._set(between_piece_x,between_piece_y,0)  # remove jumped
```

**Move validity** is also reduced to three bitwise checks (line 553 and 588):
```python
if (s & jm) and not (s & tm):
```
(The source check is implicit: we only iterate over positions that already have a peg.)

Compare to the old solver's `possible_moves()` (lines 181-196), which loops through
set members and calls `_get_square_type()` (a method doing two `in` lookups on Python
sets) for each neighbor.

**State comparison and hashing**: in the old solver, deduplication requires building a
`frozenset` of all piece coordinates (line 344):
```python
computed_states[level].add(frozenset(self.position_pieces))
```
Creating a frozenset of 30+ tuples is expensive. With the bitboard, the state IS a
Python `int` — comparison is `==` and hashing is free.

---

### 2. D4 symmetry reduction

**The idea**: the cross-shaped board has 8 symmetries (4 rotations: 0, 90, 180, 270
degrees; and 4 reflections: vertical, horizontal, main diagonal, anti-diagonal).
Two board states that are related by a symmetry are equivalent for solving purposes.
If we've already explored a state, we don't need to explore its mirror image.

**Symmetry transforms** (lines 58-77):
```python
def _rot90(x, y):
    return (_center + (_center - y), _center + (x - _center))
# ... _rot180, _rot270, _refl_v, _refl_h, _refl_d1, _refl_d2
```

**Permutation tables** (lines 79-85): for each of the 8 symmetries, a permutation
array maps each bit index to its destination after the transform:
```python
for _i, (_px, _py) in enumerate(PLAYABLE_POSITIONS):
    _nx, _ny = _transform(_px, _py)
    _perm.append(POS_TO_BIT[(_nx, _ny)])
```

**Canonical form**: the solver computes all 8 transformed versions of a state and
keeps the smallest integer as the "canonical" representative. Only canonical forms
are stored in the transposition table.

**Example**: consider a board with a single peg at position `(3,0)` (top-left of
the top arm, bit index 0). The 8 symmetries map this peg to:

| Transform | Position | Bit index |
|-----------|----------|-----------|
| Identity | (3,0) | 0 |
| Rotate 90 | (8,3) | — (in right arm) |
| Rotate 180 | (5,8) | — (in bottom arm) |
| Rotate 270 | (0,5) | — (in left arm) |
| Reflect vertical | (5,0) | 2 |
| Reflect horizontal | (3,8) | — (in bottom arm) |
| Reflect diagonal | (0,3) | — (in left arm) |
| Reflect anti-diagonal | (8,5) | — (in right arm) |

All 8 produce different integers. The canonical form is whichever is smallest. This
means a position with a lone peg in the top-left corner of the top arm is recognized
as equivalent to the corresponding position in any other arm. The solver only needs
to explore one of the 8 variants, reducing the search space by up to **8x**.

**Chunk-based fast computation** (lines 118-153): the naive `canonical_state` function
(lines 97-103) iterates over every set bit for each of the 7 non-identity symmetries.
With 30+ pegs and 7 symmetries, that is ~210+ Python loop iterations per state.

The fast version splits the 45-bit state into three 15-bit chunks:

```python
_CHUNK_SIZE = 15
_CHUNK_MASK = (1 << _CHUNK_SIZE) - 1
```

For each symmetry and each chunk, a precomputed lookup table of 2^15 = 32768 entries
maps the chunk to its transformed 45-bit output (lines 124-139). At runtime, the
canonical form is computed with just 3 table lookups per symmetry (lines 141-153):

```python
def fast_canonical_state(state):
    canon = state
    for chunk_tables in SYMMETRY_CHUNK_TABLES[1:]:
        s = state
        transformed = chunk_tables[0][s & _CHUNK_MASK]
        s >>= _CHUNK_SIZE
        transformed |= chunk_tables[1][s & _CHUNK_MASK]
        s >>= _CHUNK_SIZE
        transformed |= chunk_tables[2][s]
        if transformed < canon:
            canon = transformed
    return canon
```

This processes all 45 bits in 3 operations instead of looping bit by bit. Measured
speedup: **28x** (13.6ms vs 378ms for 10,000 calls).

---

### 3. Pagoda function pruning with chunk-based evaluation

**The idea**: a pagoda function assigns a weight `w(p)` to each position such that
for any valid move (peg at `a` jumps over `b` to land on `c`):

    w(a) + w(b) >= w(c)

This means the total weight of all pegs on the board can never increase through
valid moves. If we can find a weight assignment where the current total is less
than `w(goal)`, then the position is provably unsolvable — no sequence of moves
can ever get the total back up to the goal threshold.

**Why the old solver's pagoda was broken**: the `fast_compute_solution` (line 399-401)
set up the LP without normalization:
```python
restriction = [0 <= array_relations @ x]
objective = cp.Minimize(x @ param_pos_pieces - x @ array_final_position)
```
The trivial solution `w = 0` always satisfies these constraints with objective value 0.
Since 0 is not less than -0.0001, the solver *never found a single useful pagoda
function*. The pruning was effectively disabled.

**The fix** (line 534): adding the normalization constraint `w @ g == 1` forces the
weights to be non-trivial:
```python
constraints = [A @ w >= 0, w @ g == 1]
```

Now the LP must find weights where the goal position has weight exactly 1. This
produces useful pruning functions: the old solver found **0** functions; the new
solver finds **10** diverse functions.

**Precomputation** (lines 506-606): the solver solves the LP once per objective vector
*before* the search starts. It uses three categories of objectives:
- The initial state (all pegs except center) — finds a function that is tight on the
  starting position
- Region-concentrated states (pegs within Manhattan distance 2 of each board position)
  — finds functions sensitive to specific board areas
- Random mid-game states — finds functions that catch diverse failure modes

Each LP solve takes ~5ms. The total precomputation takes ~0.1 seconds for 10 functions.

**Example of pagoda pruning**: suppose a pagoda function assigns high weights to the
four arm tips and weight 1 to the center. After several bad moves that leave pegs
stranded in the arms but clear the center area, the total weight might drop from 8.0
to 0.7. Since 0.7 < 1.0 (the goal weight), this position is provably dead — no
sequence of moves can recover the lost weight. The solver prunes the entire subtree
without exploring it.

In a typical solve run, the pagoda functions prune over **1 million states** per
restart attempt (out of 2 million nodes checked). That is roughly **50% of all
candidates** eliminated by a simple arithmetic check.

**Chunk-based evaluation** (lines 517-526): just like symmetry, pagoda weights are
compiled into 3-chunk lookup tables. Each table maps a 15-bit chunk to the sum of
weights for the bits set in that chunk:

```python
def pagoda_prune(s):
    c0 = s & _chunk_mask
    c1 = (s >> _chunk_size) & _chunk_mask
    c2 = s >> (_chunk_size * 2)
    for chunk_tables in pagoda_chunks:
        sv = chunk_tables[0][c0] + chunk_tables[1][c1] + chunk_tables[2][c2]
        if sv < 1.0 - 1e-9:
            return True
    return False
```

Three lookups and two additions per function, instead of iterating over all ~30 set
bits per function. This makes pruning nearly free even when checking 10 functions per
candidate state.

---

### 4. Per-piece move generation with heuristic ordering

**The idea**: instead of checking all 108 directed moves against the current state
(as a global list), precompute for each position the 2-4 moves that start from it.
Then at each search node, iterate only over positions that have a peg and check their
specific moves.

**Precomputation** (lines 109-116):
```python
MOVES_BY_POS = [[] for _ in range(NUM_POSITIONS)]
for _fb, _jb, _tb, _fm, _jm, _tm, _xm in MOVES:
    MOVES_BY_POS[_fb].append((_jb, _tb, _jm, _tm, _xm))
```

**Move generation at search time** (lines 545-560):
```python
def generate_moves(s, randomize=False):
    moves = []
    temp = s
    while temp:
        bit = temp & (-temp)    # isolate lowest set bit
        pos = bit.bit_length() - 1
        for jb, tb, jm, tm, xm in _moves_by_pos[pos]:
            if (s & jm) and not (s & tm):
                noise = rng.random() * 0.5 if randomize else 0
                moves.append((-_center_dist[pos] + noise,
                              _center_dist[tb] + noise,
                              pos, jb, tb, xm))
        temp &= temp - 1       # clear lowest set bit
    moves.sort()
    return moves
```

The bit trick `temp & (-temp)` isolates the lowest set bit, and `temp &= temp - 1`
clears it. This iterates exactly over the positions that have pegs, with no wasted
checks on empty positions.

**Heuristic ordering**: moves are sorted by a two-level key:
- **Primary**: `-center_dist[source]` — prefer moving pegs that are far from center
- **Secondary**: `center_dist[destination]` — prefer landing close to center

**Example**: in the opening position, the first moves tried will be from the arm tips
(center distance 5) jumping inward, rather than shuffling pegs near the center. This
matches the known human strategy for peg solitaire: clear the arms first, then solve
the center. If we instead started by shuffling center pegs, the arm tips would become
stranded and unsolvable.

Consider an arm tip peg at `(3,0)` (distance 5 from center `(4,4)`). It can jump
over `(3,1)` to land at `(3,2)` (distance 3 from center). This move has sort key
`(-5, 3)`. Meanwhile a center-adjacent peg at `(4,3)` (distance 1) jumping to `(4,5)`
(distance 1) has sort key `(-1, 1)`. Since `-5 < -1`, the arm tip move is tried first.

---

### 5. Randomized restart DFS

**The idea**: a single deterministic DFS explores the search tree in a fixed order.
If the heuristic happens to favor a wrong branch early on, the solver can spend
millions of nodes exploring a doomed subtree before backtracking past it. Randomized
restarts avoid this by trying different orderings.

**Implementation** (lines 536-537, 554-557):
```python
for attempt in range(max_restarts):
    rng = _random.Random(attempt)
    # ...
        noise = rng.random() * 0.5 if randomize else 0
        moves.append((-_center_dist[pos] + noise,
                      _center_dist[tb] + noise,
                      pos, jb, tb, xm))
```

The first attempt (`attempt=0`) uses the pure heuristic ordering (no noise). If it
fails after 2 million nodes (line 578), the solver starts over with a different random
seed that adds small noise (0 to 0.5) to the sort keys. This keeps the overall
heuristic bias (clear periphery first) while shuffling the tie-breaking order.

Each restart gets a **fresh transposition table** (line 540):
```python
visited = set()
canon_init = _canonical(state)
visited.add(canon_init)
```

This prevents a large visited set from one failed attempt from blocking exploration
in the next attempt. The solver essentially samples different DFS paths through the
search tree.

**Example of why restarts help**: in the actual 45-hole solve, the deterministic
first attempt explores 2 million nodes and reaches depth 29/43 before its node budget
runs out. The search got stuck deep in a subtree where the last 14 pegs were arranged
in an unsolvable pattern, and backtracking through all alternatives at depth 29 is
extremely slow.

Attempt 4 (with random seed 3) happens to make a slightly different choice at depth
~10 — clearing a different arm first — which leads to a solvable late-game position.
It finds the complete 43-move solution after 1.15 million nodes in that attempt.

**The actual run** produced:
```
Attempt 1: 2000000 nodes (budget exhausted)
Attempt 2: 2000000 nodes (budget exhausted)
Attempt 3: 2000000 nodes (budget exhausted)
Attempt 4: 1153755 nodes → Yay! Found solution
Total: 7153755 nodes, 72.1 seconds
```

---

## Performance comparison

| Solver | Board | Strategy | Result |
|--------|-------|----------|--------|
| `s` (slow) | 7x7 (33 holes) | Simple backtracking | Works (slow) |
| `r` (fast) | 7x7 (33 holes) | Backtracking + runtime LP pagoda | Works (faster, but unsound) |
| `s` / `r` | 9x9 (45 holes) | Same as above | Too slow / infeasible |
| `o` (optimized) | 9x9 (45 holes) | Bitboard + symmetry + pagoda + restarts | **~70 seconds** |

### Speedup breakdown

| Optimization | What it replaces | Measured/estimated speedup |
|---|---|---|
| Bitboard (XOR moves, int state) | Python sets, frozensets, tuple creation | ~10-50x |
| Chunk-based canonical state | Bit-by-bit symmetry iteration | 28x (measured) |
| Chunk-based pagoda evaluation | Bit-by-bit weight summation | ~10x per check |
| Fixed pagoda normalization | Broken LP (0 functions found) | 0 → 10 functions (essential fix) |
| Per-piece move generation | Checking all 108 moves blindly | ~3-5x |
| Heuristic ordering (arms first) | Arbitrary move order | ~2-5x (finds solutions sooner) |
| Randomized restarts | Single deterministic DFS | Solution found on attempt 4 vs stuck |

---

## Architecture

All precomputation happens at module load time (position mapping, move tables,
symmetry chunk tables). The pagoda weights are computed once when the solver starts
(~0.1 seconds). The search itself is a tight iterative DFS loop operating purely on
integers.

The solution is translated back to `(x, y)` coordinate moves and applied to the
`PegBoard` object's history (lines 608-613), so the UI's left/right arrow replay
works as expected.
