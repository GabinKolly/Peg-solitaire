import math
import pygame
import cvxpy as cp
import numpy as np

# Defining the matrix describing the board
N = 9 # size square
width_cross = math.ceil(N/3)
width_left_part = math.ceil((N - width_cross)/2)
width_right_part = N - width_cross - width_left_part
origin = (N//2, N//2)
top_line = [-1 if x < width_left_part or x >= width_left_part + width_cross else 1 for x in range(N)]
middle_line = [1 for x in range(N)]
standard_board = [top_line.copy() if x < width_left_part or x >= width_left_part + width_cross else middle_line.copy() for x in range(N)]
standard_board[origin[1]][origin[0]] = 0

# --- Bitboard infrastructure (precomputed at module load) ---

# 2a. Position mapping: map each playable (x,y) to a bit index
PLAYABLE_POSITIONS = []
POS_TO_BIT = {}
for _y in range(N):
    for _x in range(N):
        if standard_board[_y][_x] != -1:
            POS_TO_BIT[(_x, _y)] = len(PLAYABLE_POSITIONS)
            PLAYABLE_POSITIONS.append((_x, _y))
BIT_TO_POS = PLAYABLE_POSITIONS
NUM_POSITIONS = len(PLAYABLE_POSITIONS)

# 2b. Move table: for each directed jump, precompute bitmasks
MOVES = []
for _y in range(N):
    for _x in range(N):
        if standard_board[_y][_x] == -1:
            continue
        # Horizontal right
        if _x + 2 < N and standard_board[_y][_x+1] != -1 and standard_board[_y][_x+2] != -1:
            _fb = POS_TO_BIT[(_x, _y)]
            _jb = POS_TO_BIT[(_x+1, _y)]
            _tb = POS_TO_BIT[(_x+2, _y)]
            _fm, _jm, _tm = 1 << _fb, 1 << _jb, 1 << _tb
            MOVES.append((_fb, _jb, _tb, _fm, _jm, _tm, _fm | _jm | _tm))
            # Reverse direction
            MOVES.append((_tb, _jb, _fb, _tm, _jm, _fm, _fm | _jm | _tm))
        # Vertical down
        if _y + 2 < N and standard_board[_y+1][_x] != -1 and standard_board[_y+2][_x] != -1:
            _fb = POS_TO_BIT[(_x, _y)]
            _jb = POS_TO_BIT[(_x, _y+1)]
            _tb = POS_TO_BIT[(_x, _y+2)]
            _fm, _jm, _tm = 1 << _fb, 1 << _jb, 1 << _tb
            MOVES.append((_fb, _jb, _tb, _fm, _jm, _tm, _fm | _jm | _tm))
            # Reverse direction
            MOVES.append((_tb, _jb, _fb, _tm, _jm, _fm, _fm | _jm | _tm))

# 2c. D4 symmetry permutation tables (8 symmetries of the cross board)
_center = N // 2

def _rot90(x, y):
    return (_center + (_center - y), _center + (x - _center))

def _rot180(x, y):
    return (N - 1 - x, N - 1 - y)

def _rot270(x, y):
    return (_center + (y - _center), _center + (_center - x))

def _refl_v(x, y):
    return (N - 1 - x, y)

def _refl_h(x, y):
    return (x, N - 1 - y)

def _refl_d1(x, y):
    return (y, x)

def _refl_d2(x, y):
    return (N - 1 - y, N - 1 - x)

SYMMETRY_PERMS = []
for _transform in [lambda x, y: (x, y), _rot90, _rot180, _rot270, _refl_v, _refl_h, _refl_d1, _refl_d2]:
    _perm = []
    for _i, (_px, _py) in enumerate(PLAYABLE_POSITIONS):
        _nx, _ny = _transform(_px, _py)
        _perm.append(POS_TO_BIT[(_nx, _ny)])
    SYMMETRY_PERMS.append(tuple(_perm))

def _apply_symmetry(state, perm):
    result = 0
    temp = state
    while temp:
        bit = temp & (-temp)
        src_idx = bit.bit_length() - 1
        result |= (1 << perm[src_idx])
        temp ^= bit
    return result

def canonical_state(state):
    canon = state
    for perm in SYMMETRY_PERMS[1:]:
        transformed = _apply_symmetry(state, perm)
        if transformed < canon:
            canon = transformed
    return canon

# 2d. Per-position move lookup and center distance
_center_pos = (N // 2, N // 2)
CENTER_DIST = [abs(x - _center_pos[0]) + abs(y - _center_pos[1]) for x, y in PLAYABLE_POSITIONS]

# MOVES_BY_POS[i] = list of (jumped_idx, to_idx, jumped_mask, to_mask, xor_mask)
MOVES_BY_POS = [[] for _ in range(NUM_POSITIONS)]
for _fb, _jb, _tb, _fm, _jm, _tm, _xm in MOVES:
    MOVES_BY_POS[_fb].append((_jb, _tb, _jm, _tm, _xm))

# Sort each position's moves: prefer landing closer to center
for _pos_moves in MOVES_BY_POS:
    _pos_moves.sort(key=lambda m: CENTER_DIST[m[1]])

# 2e. Chunk-based fast canonical state computation
# Split 45-bit state into 3 chunks of 15 bits each for lookup-table symmetry
_CHUNK_SIZE = 15
_NUM_CHUNKS = (NUM_POSITIONS + _CHUNK_SIZE - 1) // _CHUNK_SIZE
_CHUNK_MASK = (1 << _CHUNK_SIZE) - 1

SYMMETRY_CHUNK_TABLES = []
for _perm in SYMMETRY_PERMS:
    _chunk_tables = []
    for _chunk_idx in range(_NUM_CHUNKS):
        _start_bit = _chunk_idx * _CHUNK_SIZE
        _end_bit = min(_start_bit + _CHUNK_SIZE, NUM_POSITIONS)
        _chunk_len = _end_bit - _start_bit
        _table = [0] * (1 << _chunk_len)
        for _val in range(1 << _chunk_len):
            _result = 0
            for _bit_pos in range(_chunk_len):
                if _val & (1 << _bit_pos):
                    _result |= (1 << _perm[_start_bit + _bit_pos])
            _table[_val] = _result
        _chunk_tables.append(_table)
    SYMMETRY_CHUNK_TABLES.append(_chunk_tables)

def fast_canonical_state(state):
    """Compute canonical state using chunk lookup tables (much faster)."""
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


class PegBoard:

    def __init__(self, board_state=standard_board, goal=origin):
        self.position_pieces = set()
        self.non_playable_squares = set()
        for y, row in enumerate(board_state):
            for x, cell in enumerate(row):
                if cell==1:
                    self.position_pieces.add((x,y))
                if cell==-1:
                    self.non_playable_squares.add((x,y))
        self.goal = goal
        self.history = []
        self.width = len(board_state[0])
        self.height = len(board_state)
        self.place_in_history = -1
        self.is_current = True

    def _get_square_type(self,x,y):
        if (x,y) in self.position_pieces:
            return 1
        if (x,y) in self.non_playable_squares:
            return -1
        return 0

    def possible_moves(self):
        possible_moves = []
        for piece in self.position_pieces:
            x = piece[0]
            y = piece[1]
            if (x-1,y) in self.position_pieces:
                if x > 1 and self._get_square_type(x-2,y) == 0:
                    possible_moves.append(((x,y),(x-2,y)))
                if x < self.width-1 and self._get_square_type(x+1,y) == 0:
                    possible_moves.append(((x-1,y),(x+1,y)))
            if (x,y-1) in self.position_pieces:
                if y > 1 and self._get_square_type(x,y-2) == 0:
                    possible_moves.append(((x,y),(x,y-2)))
                if y < self.height-1 and self._get_square_type(x,y+1) == 0:
                    possible_moves.append(((x,y-1),(x,y+1)))
        return possible_moves

    
    def _set(self,x,y,square_type):
        self.position_pieces.discard((x,y))
        self.non_playable_squares.discard((x,y))
        if square_type == 1:
            self.position_pieces.add((x,y))
        if square_type == -1:
            self.non_playable_squares.add((x,y))

    def _change_goal(self,x,y):
        self.goal = (x,y)

    def get_square_type(self,x,y):
        return self._get_square_type(x,y)

    def can_move(self,x,y):
        if self._get_square_type(x,y) == 1:
            if x > 1 and self._get_square_type(x-1,y) == 1 and self._get_square_type(x-2,y) == 0:
                return True
            if x < self.width-2 and self._get_square_type(x+1,y) == 1 and self._get_square_type(x+2,y) == 0:
                return True
            if y > 1 and self._get_square_type(x,y-1) == 1 and self._get_square_type(x,y-2) == 0:
                return True
            if y < self.height-2 and self._get_square_type(x,y+1) == 1 and self._get_square_type(x,y+2) == 0:
                return True
        return False
    
    def is_over(self):
        for piece in self.position_pieces:
            if self.can_move(piece[0],piece[1]):
                return False
        return True

    def is_on_board(self,x,y):
        return x >= 0 and y >= 0 and x < self.width and y < self.height
        

    def has_won(self):
        for piece in self.position_pieces:
            if piece != self.goal:
                return False
        return True


    def _play(self,pos1,pos2,history=False):
        # Doesn't check that it is playable
        x_dif = pos2[0] - pos1[0]
        y_dif = pos2[1] - pos1[1]
        self._set(pos1[0],pos1[1],0)
        self._set(pos2[0],pos2[1],1)
        between_piece_x = pos1[0] + x_dif//2
        between_piece_y = pos1[1] + y_dif//2
        self._set(between_piece_x,between_piece_y,0)
        if history:
            if not self.is_current:
                self.history = self.history[:self.place_in_history + 1]
                self.is_current = True
            self.history.append((pos1,pos2))
            self.place_in_history += 1
        return True

    

    def play(self,pos1,pos2):
        if not self.is_on_board(pos1[0],pos1[1]) or not self.is_on_board(pos2[0],pos2[1]):
            return False
        x_dif = pos2[0] - pos1[0]
        y_dif = pos2[1] - pos1[1]
        if abs(x_dif) + abs(y_dif) != 2 or abs(x_dif) == 1:
            return False
        between_piece_x = pos1[0] + x_dif//2
        between_piece_y = pos1[1] + y_dif//2
        if self._get_square_type(pos1[0],pos1[1]) != 1 or self._get_square_type(pos2[0],pos2[1]) != 0:
            return False
        if self._get_square_type(between_piece_x,between_piece_y) != 1:
            return False
        self._set(pos1[0],pos1[1],0)
        self._set(pos2[0],pos2[1],1)
        self._set(between_piece_x,between_piece_y,0)
        if not self.is_current:
            self.history = self.history[:self.place_in_history + 1]
            self.is_current = True
        self.history.append((pos1,pos2))
        self.place_in_history += 1
        return True

    def go_back(self):
        if self.place_in_history < 0:
            return False
        last_move = self.history[self.place_in_history]
        self.place_in_history -= 1
        self.is_current = False
        pos1 = last_move[0]
        pos2 = last_move[1]
        between_piece_x = pos1[0] + (pos2[0] - pos1[0])//2
        between_piece_y = pos1[1] + (pos2[1] - pos1[1])//2
        self._set(pos1[0],pos1[1],1)
        self._set(pos2[0],pos2[1],0)
        self._set(between_piece_x,between_piece_y,1)

    def go_forward(self):
        if self.is_current:
            return False
        self.place_in_history += 1
        next_move = self.history[self.place_in_history]
        if self.place_in_history + 1 == len(self.history):
            self.is_current = True
        pos1 = next_move[0]
        pos2 = next_move[1]
        self._play(pos1,pos2)


    def compute_solution(self,screen=None,stop=None):
        # Back-tracking algorithm
        computed_states = [set() for x in range(len(self.position_pieces))]
        history_computation = [self.possible_moves()]
        level = 0 # depth in the back-tracking algorithm
        steps = [0]
        count = 0
        while level >= 0:
            count = (count + 1) %100
            if count == 0:
                if stop != None and stop():
                    return False
            if steps[level] < len(history_computation[level]):
                if steps[level] > 0 or not self.position_pieces in computed_states[level]:
                    next_move = history_computation[level][steps[level]]
                    self._play(next_move[0],next_move[1],True)
                    level += 1
                    steps[-1] += 1
                    steps.append(0)
                    history_computation.append(self.possible_moves())
                    #if screen != None:
                    #   draw_board(self,screen)
                    #   pygame.display.update()
                else:
                    steps = steps[:-1]
                    history_computation = history_computation[:-1]
                    level -= 1            
                    self.go_back()                            
            else:
                if len(history_computation[level]) == 0:
                    if self.has_won():
                        print("Yay!")
                        return True
                else:
                    computed_states[level].add(frozenset(self.position_pieces))
                steps = steps[:-1]
                history_computation = history_computation[:-1]
                level -= 1           
                self.go_back()
        self.go_forward()
        print("Nay...")
        return False

    def fast_compute_solution(self,screen,stop=None):
        """Back-tracking algorithm, with verifications with pagoda functions
        If it can prove that some position is not solvable by finding the
        right pagoda function, and there are enough pieces left, then
        it will go back two steps instead of just one. This means that
        it can sometimes miss a solution.
        """
        computed_states = [set() for x in range(len(self.position_pieces))]
        history_computation = [self.possible_moves()]
        level = 0 # depth in the back-tracking algorithm
        steps = [0]
        count = 0
        beginning_number_pieces = len(self.position_pieces)
        playable_places = []
        possible_moves = []
        position_pieces = []
        final_position = []
        position_to_index = {} # Given a position (x,y) on the board, gives the index in the list possible_moves
        for y in range(self.height):
            for x in range(self.width):
                square_type = self.get_square_type(x,y)
                if square_type != -1:
                    if square_type == 1:
                        position_pieces.append(1)
                    else:
                        position_pieces.append(0)
                    if (x,y) == self.goal:
                        final_position.append(1)
                    else:
                        final_position.append(0)
                    position_to_index.update({(x,y):len(playable_places)})
                    playable_places.append((x,y))
                    if self.is_on_board(x+2,y) and self.get_square_type(x+1,y) != -1 and self.get_square_type(x+2,y) != -1:
                        possible_moves.append(((x,y),(x+1,y),(x+2,y)))
                        possible_moves.append(((x+2,y),(x+1,y),(x,y)))
                    if self.is_on_board(x,y+2) and self.get_square_type(x,y+1) != -1 and self.get_square_type(x,y+2) != -1:
                        possible_moves.append(((x,y),(x,y+1),(x,y+2)))
                        possible_moves.append(((x,y+2),(x,y+1),(x,y)))
        n = len(playable_places)
        x = cp.Variable(n)
        matrix_relations = []
        for move in possible_moves:
            matrix_relations.append([1 if x == move[0] or x == move[1] else -1 if x == move[2] else 0 for x in playable_places])
        array_relations = np.array(matrix_relations)
        param_pos_pieces = cp.Parameter((n,),nonneg=True)
        array_final_position = np.array(final_position)
        restriction = [0 <= array_relations @ x]
        objective = cp.Minimize(x @ param_pos_pieces - x @ array_final_position)
        problem = cp.Problem(objective,restriction)
        assert problem.is_dcp(dpp=True)
        
        while level >= 0:
            count = (count + 1) %100
            if count == 0:
                if stop != None and stop():
                    return False
            pagoda_ok = True
            if steps[level] == 0 and beginning_number_pieces//3 < level and beginning_number_pieces - level > 12:
                param_pos_pieces.value = np.array(position_pieces)
                if problem.solve() < -0.0001:
                    pagoda_ok = False
            if steps[level] < len(history_computation[level]) and pagoda_ok:
                if steps[level] > 0 or not self.position_pieces in computed_states[level]:
                    next_move = history_computation[level][steps[level]]
                    self._play(next_move[0],next_move[1],True)
                    pos0 = next_move[0]
                    pos1 = next_move[1]
                    pos2 = ((pos0[0] + pos1[0])//2,(pos0[1] + pos1[1])//2)
                    position_pieces[position_to_index[pos0]] = 0
                    position_pieces[position_to_index[pos1]] = 1
                    position_pieces[position_to_index[pos2]] = 0
                    level += 1
                    steps[-1] += 1
                    steps.append(0)
                    history_computation.append(self.possible_moves())
                    #draw_board(self,screen)
                    #pygame.display.update()
                else:
                    steps = steps[:-1]
                    history_computation = history_computation[:-1]
                    level -= 1
                    prev_move = history_computation[level][steps[level]-1]
                    pos0 = prev_move[0]
                    pos1 = prev_move[1]
                    pos2 = ((pos0[0] + pos1[0])//2,(pos0[1] + pos1[1])//2)
                    position_pieces[position_to_index[pos0]] = 1
                    position_pieces[position_to_index[pos1]] = 0
                    position_pieces[position_to_index[pos2]] = 1                    
                    self.go_back()                            
            else:
                if len(history_computation[level]) == 0:
                    if self.has_won():
                        print("Yay!")
                        return True
                else:
                    computed_states[level].add(frozenset(self.position_pieces))
                m = 1 if pagoda_ok else 3
                for i in range(m):
                    if level > 0:
                        steps = steps[:-1]
                        history_computation = history_computation[:-1]
                        level -= 1
                        prev_move = history_computation[level][steps[level]-1]
                        pos0 = prev_move[0]
                        pos1 = prev_move[1]
                        pos2 = ((pos0[0] + pos1[0])//2,(pos0[1] + pos1[1])//2)
                        position_pieces[position_to_index[pos0]] = 1
                        position_pieces[position_to_index[pos1]] = 0
                        position_pieces[position_to_index[pos2]] = 1                 
                        self.go_back()
        self.go_forward()
        print("Nay...")
        return False

    def pagoda_fun_test(self):
        playable_places = []
        possible_moves = []
        position_pieces = []
        final_position = []
        for x in range(self.width):
            for y in range(self.height):
                square_type = self.get_square_type(x,y)
                if square_type != -1:
                    if square_type == 1:
                        position_pieces.append(1)
                    else:
                        position_pieces.append(0)
                    if (x,y) == self.goal:
                        final_position.append(1)
                    else:
                        final_position.append(0)
                    playable_places.append((x,y))
                    if self.is_on_board(x+2,y) and self.get_square_type(x+1,y) != -1 and self.get_square_type(x+2,y) != -1:
                        possible_moves.append(((x,y),(x+1,y),(x+2,y)))
                        possible_moves.append(((x+2,y),(x+1,y),(x,y)))
                    if self.is_on_board(x,y+2) and self.get_square_type(x,y+1) != -1 and self.get_square_type(x,y+2) != -1:
                        possible_moves.append(((x,y),(x,y+1),(x,y+2)))
                        possible_moves.append(((x,y+2),(x,y+1),(x,y)))
        n = len(playable_places)
        x = cp.Variable(n)
        matrix_relations = []
        for move in possible_moves:
            matrix_relations.append([1 if x == move[0] or x == move[1] else -1 if x == move[2] else 0 for x in playable_places])
        array_relations = np.array(matrix_relations)
        array_position_pieces = np.array(position_pieces)
        array_final_position = np.array(final_position)
        restriction = [0 <= array_relations @ x, 1 == x @ final_position]
        objective = cp.Minimize(x @ array_position_pieces - x @ array_final_position)
        problem = cp.Problem(objective,restriction)
        if problem.solve() < -0.0001:
            return False
        return True

    @staticmethod
    def _precompute_pagoda_weights(goal_bit_index, max_functions=10):
        """Precompute pagoda function weight vectors for pruning.

        Solves the LP with normalization w@g==1 so the trivial w=0 is excluded.
        Each weight vector can independently prove a position unsolvable:
        if sum(w[i] for i in pegs) < 1, the position cannot reach the goal.

        Returns chunk-based lookup tables for fast evaluation.
        """
        n = NUM_POSITIONS
        rows = []
        seen_moves = set()
        for fb, jb, tb, _, _, _, _ in MOVES:
            key = (fb, jb, tb)
            if key not in seen_moves:
                seen_moves.add(key)
                row = [0] * n
                row[fb] += 1
                row[jb] += 1
                row[tb] -= 1
                rows.append(row)

        A = np.array(rows, dtype=float)
        g = np.zeros(n)
        g[goal_bit_index] = 1.0

        w = cp.Variable(n)
        constraints = [A @ w >= 0, w @ g == 1]

        raw_weights = []
        rng = np.random.RandomState(42)

        objectives = []
        # Initial state
        s_init = np.ones(n)
        s_init[goal_bit_index] = 0.0
        objectives.append(s_init)

        # Region-concentrated states
        for region_center_idx in range(n):
            s = np.zeros(n)
            for j in range(n):
                px, py = PLAYABLE_POSITIONS[j]
                rx, ry = PLAYABLE_POSITIONS[region_center_idx]
                if abs(px - rx) + abs(py - ry) <= 2:
                    s[j] = 1.0
            objectives.append(s)

        # Random mid-game states
        for _ in range(20):
            s = np.zeros(n)
            num_pegs = rng.randint(3, n - 3)
            peg_positions = rng.choice(n, size=num_pegs, replace=False)
            s[peg_positions] = 1.0
            objectives.append(s)

        param_s = cp.Parameter(n, nonneg=True)
        objective = cp.Minimize(w @ param_s - 1)
        problem = cp.Problem(objective, constraints)

        seen_hashes = set()
        for s_vec in objectives:
            param_s.value = s_vec
            try:
                problem.solve(warm_start=True)
                if w.value is not None:
                    weights = w.value.copy()
                    w_hash = tuple(round(v, 2) for v in weights)
                    if w_hash not in seen_hashes:
                        seen_hashes.add(w_hash)
                        raw_weights.append(weights)
            except Exception:
                pass

        # Keep only the most diverse functions (up to max_functions)
        if len(raw_weights) > max_functions:
            raw_weights = raw_weights[:max_functions]

        # Build chunk-based lookup tables for fast pagoda evaluation
        # For each function k, PAGODA_CHUNKS[k][chunk_idx][chunk_val] = sum of weights
        chunk_size = _CHUNK_SIZE
        num_chunks = _NUM_CHUNKS
        pagoda_chunk_tables = []
        for weights in raw_weights:
            chunk_tables = []
            for chunk_idx in range(num_chunks):
                start_bit = chunk_idx * chunk_size
                end_bit = min(start_bit + chunk_size, n)
                chunk_len = end_bit - start_bit
                table = [0.0] * (1 << chunk_len)
                for val in range(1 << chunk_len):
                    s = 0.0
                    for bit_pos in range(chunk_len):
                        if val & (1 << bit_pos):
                            s += weights[start_bit + bit_pos]
                    table[val] = s
                chunk_tables.append(table)
            pagoda_chunk_tables.append(chunk_tables)

        return pagoda_chunk_tables

    def _apply_bitboard_solution(self, solution_moves):
        """Convert bitboard solution moves to UI format and apply."""
        for fb, jb, tb in solution_moves:
            from_pos = BIT_TO_POS[fb]
            to_pos = BIT_TO_POS[tb]
            self._play(from_pos, to_pos, True)

    def optimized_compute_solution(self, screen=None, stop=None):
        """Optimized solver using bitboard, chunk-based symmetry/pagoda, and restarts.

        Key optimizations over previous version:
        - Chunk lookup tables for canonical state (3 lookups vs ~240 bit iterations)
        - Chunk lookup tables for pagoda evaluation (3 lookups vs ~30 bit iterations)
        - Randomized restart DFS to explore diverse paths
        - Per-piece move generation with arm-clearing heuristic
        """
        import time as _time
        import random as _random

        # Convert current board to bitboard
        state = 0
        for (x, y) in self.position_pieces:
            if (x, y) in POS_TO_BIT:
                state |= (1 << POS_TO_BIT[(x, y)])

        goal_bit = POS_TO_BIT[self.goal]
        goal_state = 1 << goal_bit
        num_initial_pegs = bin(state).count('1')

        if state == goal_state:
            print("Yay!")
            return True

        # Precompute pagoda weights with chunk tables
        print("Precomputing pagoda functions...")
        pagoda_chunks = self._precompute_pagoda_weights(goal_bit)
        print(f"Got {len(pagoda_chunks)} pagoda functions for pruning.")

        # Local references for performance
        _moves_by_pos = MOVES_BY_POS
        _center_dist = CENTER_DIST
        _canonical = fast_canonical_state
        _chunk_mask = _CHUNK_MASK
        _chunk_size = _CHUNK_SIZE

        def pagoda_prune(s):
            """Return True if state s is provably unsolvable (chunk-based)."""
            c0 = s & _chunk_mask
            c1 = (s >> _chunk_size) & _chunk_mask
            c2 = s >> (_chunk_size * 2)
            for chunk_tables in pagoda_chunks:
                sv = chunk_tables[0][c0] + chunk_tables[1][c1] + chunk_tables[2][c2]
                if sv < 1.0 - 1e-9:
                    return True
            return False

        total_nodes = 0
        total_pruned = 0
        start_time = _time.time()
        nodes_per_restart = 2_000_000
        max_restarts = 500

        print("Searching with randomized restarts...")

        for attempt in range(max_restarts):
            rng = _random.Random(attempt)

            # Transposition table per restart (fresh each time to avoid over-constraining)
            visited = set()
            canon_init = _canonical(state)
            visited.add(canon_init)
            max_visited = 5_000_000

            def generate_moves(s, randomize=False):
                """Generate valid moves sorted by heuristic with optional randomization."""
                moves = []
                temp = s
                while temp:
                    bit = temp & (-temp)
                    pos = bit.bit_length() - 1
                    for jb, tb, jm, tm, xm in _moves_by_pos[pos]:
                        if (s & jm) and not (s & tm):
                            noise = rng.random() * 0.5 if randomize else 0
                            moves.append((-_center_dist[pos] + noise,
                                          _center_dist[tb] + noise,
                                          pos, jb, tb, xm))
                    temp &= temp - 1
                moves.sort()
                return moves

            initial_moves = generate_moves(state, randomize=(attempt > 0))
            stack = [(state, initial_moves, 0)]
            solution_path = []
            nodes = 0
            pruned = 0

            while stack:
                nodes += 1
                if nodes & 0x3FFF == 0:
                    if stop is not None and stop():
                        total_nodes += nodes
                        total_pruned += pruned
                        elapsed = _time.time() - start_time
                        print(f"Stopped after {total_nodes} nodes total, {elapsed:.1f}s")
                        return False

                if nodes >= nodes_per_restart:
                    break

                cur_state, cur_moves, mi = stack[-1]

                found_move = False
                while mi < len(cur_moves):
                    _, _, fb, jb, tb, xm = cur_moves[mi]
                    mi += 1

                    new_state = cur_state ^ xm

                    # Win check
                    if new_state == goal_state:
                        solution_path.append((fb, jb, tb))
                        self._apply_bitboard_solution(solution_path)
                        total_nodes += nodes
                        elapsed = _time.time() - start_time
                        print(f"Yay! Found in {total_nodes} nodes, "
                              f"attempt {attempt+1}, {elapsed:.1f}s")
                        return True

                    # Transposition check (chunk-based canonical)
                    canon = _canonical(new_state)
                    if canon in visited:
                        continue

                    # Pagoda pruning (chunk-based, mid-game only)
                    pc = bin(new_state).count('1')
                    if 5 <= pc <= 35 and pagoda_chunks:
                        if pagoda_prune(new_state):
                            pruned += 1
                            continue

                    # Accept move
                    if len(visited) < max_visited:
                        visited.add(canon)
                    solution_path.append((fb, jb, tb))
                    stack[-1] = (cur_state, cur_moves, mi)
                    new_moves = generate_moves(new_state, randomize=(attempt > 0))
                    stack.append((new_state, new_moves, 0))
                    found_move = True
                    break

                if not found_move:
                    stack.pop()
                    if solution_path:
                        solution_path.pop()

            total_nodes += nodes
            total_pruned += pruned
            if nodes & 0xFFFFF == 0 or attempt % 10 == 0:
                elapsed = _time.time() - start_time
                depth = len(solution_path) if solution_path else 0
                print(f"  Attempt {attempt+1}: {nodes} nodes, max depth reached, "
                      f"{total_nodes} total, {total_pruned} pruned, {elapsed:.1f}s")

        elapsed = _time.time() - start_time
        print(f"Nay... {total_nodes} nodes across {max_restarts} attempts, {elapsed:.1f}s")
        return False


def draw_board(board,screen,selected_square=None,godmode=False):
    screen.fill((255,255,255))
    (x,y) = screen.get_size()
    size_row = board.width
    size_column = board.height
    size_square = x/size_row
    goal = board.goal
    if godmode:
        color_piece = (0,255,0) # green
    else:
        color_piece = (0,36,215) #blue
    color_selected_piece = (255,128,0) #pink
    for i in range(size_column + 1):
        pygame.draw.line(screen,(0,0,0),(0,i*size_square),(x,i*size_square),2)
    for j in range(size_row + 1):
        pygame.draw.line(screen,(0,0,0),(j*size_square,0),(j*size_square,y),2)
    for i in range(size_column):
        for j in range(size_row):
            if board.get_square_type(j,i) == 1:
                pygame.draw.circle(screen,color_piece,((j + 0.5) * size_square + 1,(i + 0.5) * size_square + 1),size_square/2.5)
            elif board.get_square_type(j,i) == -1:
                screen.fill((0,0,0),pygame.Rect(j*size_square,i*size_square,size_square,size_square))
    if not selected_square == None:
        pygame.draw.circle(screen,color_selected_piece,((selected_square[0] + 0.5) * size_square + 1,(selected_square[1] + 0.5) * size_square + 1),size_square/2.5)
    pygame.draw.rect(screen,(255,0,0),pygame.Rect(goal[0] * size_square, goal[1] * size_square,size_square + 2,size_square + 2),2)



if __name__ == '__main__':
    board = PegBoard()
    size_square = 35
    length_screen = size_square * board.width
    height_screen = size_square * board.height
    selected_square = None
    screen = pygame.display.set_mode((length_screen,height_screen))
    pygame.init()
    draw_board(board,screen,selected_square)
    pygame.display.update()
    godmode = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x = pos[0] // size_square
                y = pos[1] // size_square
                if godmode:
                    square_type = board.get_square_type(x,y)
                    if event.button == 1:
                        square_type += 1
                        if square_type == 2:
                            square_type = -1
                        board._set(x,y,square_type)
                    else:
                        if square_type != -1:
                            board._change_goal(x,y)
                else:
                    if selected_square == None:
                        if board.get_square_type(x,y) == 1:
                            selected_square = (x,y)
                    else:
                        if abs(selected_square[0] - x) + abs(selected_square[1] - y) == 2 and abs(selected_square[0] - x) != 1:
                            board.play(selected_square,(x,y))
                        selected_square = None
                    
                draw_board(board,screen,selected_square,godmode)
                pygame.display.update()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    godmode = not godmode
                    selected_square = None
                if  event.key == pygame.K_LEFT:
                    selected_square = None
                    board.go_back()
                if event.key == pygame.K_RIGHT:
                    selected_square = None
                    board.go_forward()
                if event.key == pygame.K_s:
                    def stop():
                        for eventt in pygame.event.get():
                            if eventt.type == pygame.KEYDOWN and eventt.key == pygame.K_s:
                                return True
                        return False
                    board.compute_solution(screen,stop)
                if event.key == pygame.K_r:
                    def stop():
                        for eventt in pygame.event.get():
                            if eventt.type == pygame.KEYDOWN and eventt.key == pygame.K_r:
                                return True
                        return False
                    board.fast_compute_solution(screen,stop)
                if event.key == pygame.K_o:
                    def stop():
                        for eventt in pygame.event.get():
                            if eventt.type == pygame.KEYDOWN and eventt.key == pygame.K_o:
                                return True
                        return False
                    board.optimized_compute_solution(screen,stop)

                if event.key == pygame.K_t:
                    print(board.pagoda_fun_test())
                    
                draw_board(board,screen,selected_square,godmode)
                pygame.display.update()