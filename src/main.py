from ai.player import Player
from ai.train import train
from game.tictactoe import TicTacToe, TEST_DATA

def test(game, playerType, testData):
  player = playerType(game)
  print("-- Untrained --")
  for turn, moves, score in testData:
    turn = turn.clone()
    print(player.predict([game.inputs()]))
  i = 0
  for x in range(1000):
    train(game, player, rng_prob = 1 / (x + 1))
    player.save_weights('./checkpoints/tictactoe')
    print("-- Trained " + str(i) + " --")
    i += 1
    loss = 0
    for turn, moves, score in testData:
      turn = turn.clone()
      mps, scores = player.predict([turn.inputs()])
      print(turn.inputs())
      print(str(turn) + ":" + ", ".join("%0.2f" % mp for mp in mps[0]) + ":" + ", ".join("%0.2f" % s for s in scores[0]))

if __name__ == "__main__":
  test(TicTacToe(), Player, TEST_DATA)
