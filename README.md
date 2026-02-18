# Peg solitaire
## How to use
An implementation of Peg solitaire in Python. The default board is a 9x9 cross
with 45 holes (change `N` in the code to use other sizes, e.g. `N = 7` for the
classic 33-hole English board). You can also create custom boards in god mode.

### Controls
- **Click** a peg, then click a destination to make a move
- **Left/Right arrows**: undo/redo moves
- **g**: toggle god mode (edit the board freely; right-click to set the goal)
- **s**: compute a solution using the slow backtracking solver
- **r**: compute a solution using the fast solver (pagoda pruning, may miss solutions)
- **o**: compute a solution using the **optimized solver** (recommended for the 45-hole board)
- **t**: test if a pagoda function proves the current position has no solution
- Press the same key again (s/r/o) to stop a running solver

## Solvers

### Slow solver (s)
A simple back-tracking algorithm. It explores the game tree depth-first and avoids
revisiting identical board positions at the same depth level. Works for small boards
but is too slow for the 45-hole cross.

### Fast solver (r)
A back-tracking algorithm that uses pagoda functions to prune dead branches. At each
step it solves a linear program (via `cvxpy`) to check if the current position is
provably unsolvable. If it finds a proof, it backtracks multiple levels. This works
surprisingly well on small boards but is still too slow for 45 holes because the LP
is solved at runtime during the search.

### Optimized solver (o)
A heavily optimized solver designed for the 45-hole cross board. Key techniques:
- **Bitboard representation**: board state as a single integer, moves via XOR
- **D4 symmetry reduction**: 8 symmetries with chunk lookup tables (28x faster)
- **Precomputed pagoda pruning**: pagoda weights computed once before search, evaluated via chunk lookups
- **Heuristic move ordering**: clears periphery first, then center
- **Randomized restart DFS**: multiple attempts with varied move ordering

Solves the 45-hole board in about 70 seconds. See [OPTIMIZED_SOLVER.md](OPTIMIZED_SOLVER.md)
for a detailed explanation.

## Pagoda functions
A pagoda function is a function f that associates to each position a real number, such that
for any three consecutive positions a, b, c, we have f(a) + f(b) >= f(c). It means that the
sum of values of f at each piece cannot increase by doing valid moves, and so if the value of
f at the goal is greater than this sum, it means that there is no solution from this situation.
To find these pagoda functions, we use a linear optimization module (`cvxpy`).
