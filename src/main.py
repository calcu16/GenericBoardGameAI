from ai.player import Player
from ai.train import train
from game.tictactoe import TicTacToe
from sys import argv
from argparse import ArgumentParser

def cmd_train(game, playerType, args):
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
    for turn, moves, score in game.test_data:
      turn = turn.clone()
      mps, scores = next(zip(*player.predict([turn.inputs()])))
      wprob = sum(mps[move] for move in moves)
      dscore = (scores[0] - score) ** 2
      if wprob < 0.9 or dscore > 0.05**2:
        bad = True
        print(str(turn) + ":" + ", ".join("%0.2f" % mp for mp in mps) + ":" + ", ".join("%0.2f" % s for s in scores))

def cmd_evaluate(game, playerType, args):
  player = playerType(game)
  if args.load:
    player.load_weights('./checkpoints/tictactoe')
  game = game.clone()
  ap = game.activePlayer()
  while game.activePlayer() is not None:
    inp = game.inputs()
    mp, eval = player.predict([inp])
    p, m = max((p, i) for (i, p) in enumerate(mp[0]))
    game.move(m)
    print("%d %0.2f %0.2f" % (m, p, eval[0][ap]))
  print(game.winner()[ap])

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--action', default='evaluate')
  parser.add_argument('--load', action="store_true")
  parser.add_argument('--k', type=int, default=1)
  parser.add_argument('--position', default="")
  args = parser.parse_args()

  moves = [int(m) for m in args.position.split()]
  if args.action == 'train':
    cmd_train(TicTacToe(moves), Player, args)
  elif args.action == 'evaluate':
    cmd_evaluate(TicTacToe(moves), Player, args)
