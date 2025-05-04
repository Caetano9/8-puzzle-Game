from collections import deque
import heapq
import time

# Dimensão do tabuleiro
N = 3

# Vetores de deslocamento: ← → ↑ ↓
ROW = (0, 0, -1, 1)
COL = (-1, 1, 0, 0)

# Estrutura de estado
class PuzzleState:
    __slots__ = ("board", "x", "y", "depth", "cost")

    def __init__(self, board, x, y, depth, cost=0):
        self.board = board
        self.x = x
        self.y = y
        self.depth = depth        # g(n)
        self.cost = cost          # f(n) para A*

    def __lt__(self, other):
        return self.cost < other.cost

# Funções utilitárias
GOAL_BOARD = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))

def is_goal_state(board):
    return tuple(map(tuple, board)) == GOAL_BOARD

def is_valid(x, y):
    return 0 <= x < N and 0 <= y < N

def board_to_tuple(board):
    return tuple(tuple(r) for r in board)

def print_board(board):
    for r in board:
        print(" ".join(" " if n == 0 else str(n) for n in r))
    print()

# Heurísticas para o A*
def h_manhattan(board):
    dist = 0
    for i in range(N):
        for j in range(N):
            v = board[i][j]
            if v:
                gi, gj = divmod(v - 1, N)
                dist += abs(i - gi) + abs(j - gj)
    return dist

def h_misplaced(board):
    return sum(1 for i in range(N) for j in range(N)
               if board[i][j] and board[i][j] != GOAL_BOARD[i][j])

# Busca em profundidade limitada (DFS)
def solve_dfs(start, x, y, depth_limit=50):
    stack = [PuzzleState(start, x, y, 0)]
    parent = {board_to_tuple(start): None}
    visited = {board_to_tuple(start)}
    expanded = 0

    while stack:
        curr = stack.pop()
        expanded += 1
        if is_goal_state(curr.board):
            return parent, curr, expanded
        if curr.depth >= depth_limit:
            continue
        for k in range(4):
            nx, ny = curr.x + ROW[k], curr.y + COL[k]
            if is_valid(nx, ny):
                new_board = [r[:] for r in curr.board]
                new_board[curr.x][curr.y], new_board[nx][ny] = \
                    new_board[nx][ny], new_board[curr.x][curr.y]
                key = board_to_tuple(new_board)
                if key not in visited:
                    visited.add(key)
                    ns = PuzzleState(new_board, nx, ny, curr.depth + 1)
                    parent[key] = curr
                    stack.append(ns)
    return None, None, expanded

# Busca em largura (BFS)
def solve_bfs(start, x, y):
    q = deque([PuzzleState(start, x, y, 0)])
    parent = {board_to_tuple(start): None}
    visited = {board_to_tuple(start)}
    expanded = 0

    while q:
        curr = q.popleft()
        expanded += 1
        if is_goal_state(curr.board):
            return parent, curr, expanded
        for k in range(4):
            nx, ny = curr.x + ROW[k], curr.y + COL[k]
            if is_valid(nx, ny):
                new_board = [r[:] for r in curr.board]
                new_board[curr.x][curr.y], new_board[nx][ny] = \
                    new_board[nx][ny], new_board[curr.x][curr.y]
                key = board_to_tuple(new_board)
                if key not in visited:
                    visited.add(key)
                    ns = PuzzleState(new_board, nx, ny, curr.depth + 1)
                    parent[key] = curr
                    q.append(ns)
    return None, None, expanded

# A* genérico
def solve_astar(start, x, y, h_func):
    open_heap = []
    parent = {board_to_tuple(start): None}
    g_cost = {board_to_tuple(start): 0}
    start_state = PuzzleState(start, x, y, 0, h_func(start))
    heapq.heappush(open_heap, start_state)
    expanded = 0

    while open_heap:
        curr = heapq.heappop(open_heap)
        expanded += 1
        if is_goal_state(curr.board):
            return parent, curr, expanded
        for k in range(4):
            nx, ny = curr.x + ROW[k], curr.y + COL[k]
            if is_valid(nx, ny):
                new_board = [r[:] for r in curr.board]
                new_board[curr.x][curr.y], new_board[nx][ny] = \
                    new_board[nx][ny], new_board[curr.x][curr.y]
                ntuple = board_to_tuple(new_board)
                tentative_g = curr.depth + 1
                if ntuple not in g_cost or tentative_g < g_cost[ntuple]:
                    g_cost[ntuple] = tentative_g
                    ns = PuzzleState(new_board, nx, ny,
                                     tentative_g,
                                     tentative_g + h_func(new_board))
                    parent[ntuple] = curr
                    heapq.heappush(open_heap, ns)
    return None, None, expanded

# Reconstrução e saída
def reconstruct(parent, node):
    path = []
    while node:
        path.append(node)
        node = parent[board_to_tuple(node.board)]
    return list(reversed(path))

def show_path(path):
    for step in path:
        print(f"Profundidade: {step.depth}")
        print_board(step.board)

def run_and_report(name, solver, *args, **kwargs):
    t0 = time.perf_counter()
    par, goal, expanded = solver(*args, **kwargs)
    dt = time.perf_counter() - t0
    if goal:
        print(f"{name:13} | tempo = {dt:7.4f} s | nós expandidos = {expanded:6d} "
              f"| profundidade da solução = {goal.depth:2d}")
    else:
        print(f"{name:13} | não encontrou solução no limite dado.")

# Programa principal
if __name__ == "__main__":
    start_board = [[1, 2, 3],
                   [4, 0, 5],
                   [6, 7, 8]]
    x0, y0 = 1, 1  # posição inicial do branco

    print("Estado inicial:")
    print_board(start_board)

    run_and_report("DFS limit=50", solve_dfs, start_board, x0, y0, depth_limit=50)
    run_and_report("BFS",          solve_bfs, start_board, x0, y0)
    run_and_report("A* Manhattan", solve_astar, start_board, x0, y0, h_manhattan)
    run_and_report("A* Misplaced", solve_astar, start_board, x0, y0, h_misplaced)

    par, goal, _ = solve_astar(start_board, x0, y0, h_manhattan)
    path = reconstruct(par, goal)
    show_path(path)

    # Mantém a janela do console aberta até o usuário confirmar
    input("Pressione Enter para sair...")