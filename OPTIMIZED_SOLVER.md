# Optimized Solver for the 45-Hole Cross Board

The optimized solver (press `o`) can solve the 9x9 cross-shaped board with 45 holes
in about 70 seconds. This document explains how it works and why the previous solvers
(`s` and `r`) are too slow for this board size.

## The problem

The 45-hole cross board has 44 initial pegs (all holes filled except center) and
requires 43 moves to reach the goal of a single peg at the center. The search tree
is enormous: each position can have 4-10 valid moves, leading to a branching factor
that makes naive backtracking infeasible.

## Key optimizations

### 1. Bitboard representation

Instead of storing piece positions as a Python `set` of `(x, y)` tuples, the board
state is encoded as a single 45-bit integer. Each of the 45 playable positions is
assigned a bit index (0-44).

- **Move application**: A single XOR operation (`state ^= xor_mask`) simultaneously
  removes the source peg, removes the jumped peg, and places the destination peg.
  Undoing a move is the same XOR.
- **State comparison/hashing**: Comparing two states is just integer comparison (`==`).
  The integer itself serves as its own hash.
- **Move validity**: Three bitwise checks:
  `(state & from_mask) and (state & jumped_mask) and not (state & to_mask)`

### 2. D4 symmetry reduction

The cross-shaped board has 8 symmetries (4 rotations and 4 reflections). Two board
states that are symmetric are equivalent for solving purposes. The solver computes
a *canonical form* for each state (the smallest integer among all 8 symmetric
variants) and stores only canonical forms in the transposition table. This reduces
the effective search space by up to 8x.

To make this fast, the 45-bit state is split into three 15-bit chunks. For each
symmetry and each chunk, a precomputed lookup table (32768 entries) maps the chunk
to its transformed bits. Computing the canonical state requires only 7 x 3 = 21
table lookups instead of iterating over every set bit. This is **28x faster** than
the naive bit-by-bit approach.

### 3. Pagoda function pruning with chunk-based evaluation

A pagoda function assigns a weight `w(p)` to each position `p` such that for any
valid move (peg at `a` jumps over `b` to land on `c`):

    w(a) + w(b) >= w(c)

This means the total weight of all pegs on the board can never increase. If the
total weight drops below `w(goal)`, the position is provably unsolvable.

**Precomputation**: Before the search begins, the solver finds 10 diverse pagoda
weight vectors by solving a linear program (via `cvxpy`) with different objective
functions. The LP includes a normalization constraint (`w @ goal == 1`) to avoid
the trivial zero solution.

**Fast evaluation**: Each pagoda function's weights are compiled into chunk-based
lookup tables (same 3x15-bit split as symmetry). Evaluating a pagoda function on
a state requires just 3 table lookups and 2 additions, instead of iterating over
all set bits.

### 4. Per-piece move generation with heuristic ordering

Instead of checking all 108 directed moves against the current state, the solver
uses precomputed per-position move lists (`MOVES_BY_POS[i]`). It iterates only
over positions that have a peg and checks their 2-4 possible moves.

Moves are sorted by a heuristic that clears the board periphery first:
- **Primary key**: prefer moving pegs that are *far from center* (negative center
  distance of the source)
- **Secondary key**: prefer landing *close to center* (center distance of the
  destination)

This "clear arms first, then center" strategy matches known human strategies for
solving peg solitaire.

### 5. Randomized restart DFS

A single deterministic DFS can get stuck exploring a large fruitless subtree. The
solver uses **randomized restarts**: after exploring 2 million nodes without finding
a solution, it restarts the search with a different random perturbation of the move
ordering. The first attempt uses the pure heuristic order; subsequent attempts add
small random noise to the sort keys, causing the DFS to explore different paths.

Each restart uses a fresh transposition table, so previously explored states don't
block new paths.

## Performance comparison

| Solver | Board | Strategy | Result |
|--------|-------|----------|--------|
| `s` (slow) | 7x7 (33 holes) | Simple backtracking | Works (slow) |
| `r` (fast) | 7x7 (33 holes) | Backtracking + runtime LP pagoda | Works (faster, but unsound) |
| `s` / `r` | 9x9 (45 holes) | Same as above | Too slow / infeasible |
| `o` (optimized) | 9x9 (45 holes) | Bitboard + symmetry + pagoda + restarts | **~70 seconds** |

## Architecture

All precomputation happens at module load time (position mapping, move tables,
symmetry chunk tables). The pagoda weights are computed once when the solver starts.
The search itself is a tight iterative DFS loop operating purely on integers.

The solution is translated back to `(x, y)` coordinate moves and applied to the
`PegBoard` object's history, so the UI's left/right arrow replay works as expected.
