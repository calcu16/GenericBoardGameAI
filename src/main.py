from ai.player import Player
from ai.train import train
from game.tictactoe import TicTacToe, TEST_DATA

def test(game, playerType, testData):
  player = playerType(game)
  print("-- Untrained --")
  for turn, moves, score in testData:
    turn = turn.clone()
    print(player.predict([game.inputs()]))
  loss = 1.0
  i = 0
  while loss >= 0.1:
    train(game, player)
    player.save_weights('./checkpoints/tictactoe')
    print("-- Trained " + str(i) + " --")
    i += 1
    loss = 0
    for turn, moves, score in testData:
      turn = turn.clone()
      mps, scores = player.predict([game.inputs()])
      loss = 1 - sum(mps[0][move] for move in moves)
      print(turn)
      print(mps[0])
      print(scores[0])
      print(loss)

if __name__ == "__main__":
  test(TicTacToe(), Player, TEST_DATA)
