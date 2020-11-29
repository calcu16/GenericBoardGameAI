def _wins():
  for i in range(3):
    yield tuple(3*i + j for j in range(3))
    yield tuple(i + 3*j for j in range(3))
  yield tuple(4*j for j in range(3))
  yield tuple(2*(j+1) for j in range(3))
WINS = list(_wins())

_TEST_DATA = []

class TicTacToe:
  def __init__(self, moves = []):
    global _TEST_DATA
    self.num_inputs = 9 * 2
    self.num_actions = 9
    self.num_players = 2
    self.moves = moves.copy()
    self.test_data = _TEST_DATA
  def clone(self):
    return TicTacToe(self.moves)
  def inputs(self):
    inp = [[0] * 9] * 2
    for i, m in enumerate(self.moves):
      p = i % 2
      inp[p][m] = 1
    return [x for xs in inp for x in xs]
  def activePlayer(self):
    return len(self.moves) % 2 if len(self.moves) < 9 and self.winner()[0] == 0.5 else None
  def move(self, m):
    if m >= 9:
      raise Exception("Bad move: %d\n" % m)
    self.moves.append(m)
  def winner(self):
    global WINS
    ms = (set(), set())
    for i, m in enumerate(self.moves):
      p = i % 2
      q = 1 - p
      if m in ms[q] or m in ms[p]:
        return { p : -1.0, q : 1.0 } # illegal move
      ms[p].add(m)
      for win in WINS:
        if all(w in ms[p] for w in win):
          return { p : 1.0, q : 0.0 }
    return { 0 : 0.5, 1 : 0.5 }
  def __str__(self): return str(self.moves)

_TEST_DATA.extend([
  (TicTacToe([4, 6, 3, 5, 7, 1, 2, 8]), set([0]), 0.5),
  (TicTacToe([4, 2, 8]), set([0]), 0.5),
  (TicTacToe([4, 2, 8, 1]), set([0]), 1.0),
])

