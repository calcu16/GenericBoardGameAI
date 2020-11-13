from ai.player import Player
from ai.train import train
from game.tictactoe import TicTacToe, TEST_DATA
from sys import argv
from argparse import ArgumentParser

def test(game, playerType, testData, args):
  player = playerType(game)
  if args.load:
    player.load_weights('./checkpoints/tictactoe')
  i = 0
  bad = True
  while bad:
    train(game, player, rng_prob = 1.0 / (i + args.k))
    player.save_weights('./checkpoints/tictactoe')
    print("-- Trained " + str(i) + " --")
    i += 1
    loss = 0
    bad = False
    for turn, moves, score in testData:
      turn = turn.clone()
      mps, scores = next(zip(*player.predict([turn.inputs()])))
      wprob = sum(mps[move] for move in moves)
      dscore = (scores[0] - score) ** 2
      if wprob < 0.9 or dscore > 0.05**2:
        bad = True
        print(str(turn) + ":" + ", ".join("%0.2f" % mp for mp in mps) + ":" + ", ".join("%0.2f" % s for s in scores))

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--load', action="store_true")
  parser.add_argument('--k', type=int, default=1)
  args = parser.parse_args()
  test(TicTacToe(), Player, TEST_DATA, args)
