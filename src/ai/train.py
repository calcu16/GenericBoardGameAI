from random import Random
import tensorflow as tf

class Turn:
  def __init__(self, inputs, player, move, scores):
    self.inputs = inputs
    self.player = player
    self.move = move
    self.bwp = scores[player]
    self.ewp = None
  def updateWinner(self, winners):
    if self.player in winners:
      self.ewp = 1.0 / len(winners)
      return True
    else:
      self.ewp = 0.0
      return False
  def updateScore(self, scores):
    self.ewp = scores[self.player]
  def deltaScore(self):
    return self.ewp - self.bwp

class Game:
  def __init__(self, rng, game):
    self.game = game.clone()
    self.rng = rng
    self.pi = None
    self.lturn = None
  def activePlayer(self):
    self.pi = self.game.activePlayer()
    return self.pi is not None
  def move(self, inp, mp, scores):
    if self.lturn is not None:
      self.lturn.updateScore(scores)
    m = self.rng.choices(range(len(mp)), weights=mp)[0]
    self.lturn = Turn(inp, self.pi, m, scores)
    self.game.move(m)
    return self.lturn
  def updateWinner(self):
    self.lturn.updateWinner(self.game.winner())

OPTIMIZER = tf.keras.optimizers.Adam()
def train(game, player, num_games = 32, num_training_epochs = 5):
  global OPTIMIZER
  rng = Random()
  players = [player for i in range(game.num_players)]
  turns = []
  gs = [Game(rng, game) for i in range(num_games)]
  while gs:
    pgs = [(player, []) for player in players]
    for g in gs:
      if g.activePlayer():
        pgs[g.pi][1].append(g)
      else:
        g.updateWinner()
    for (p, gs) in pgs:
      if not gs: continue
      inp = [g.game.inputs() for g in gs]
      results = player.predict(inp)
      for g, inp, (mp, scores) in zip(gs, inp, zip(*results)):
        turns.append(g.move(inp, mp, scores))
    gs = [g for p,gs in pgs for g in gs]
   
  for i in range(num_training_epochs):
    dataset = tf.data.Dataset.from_tensor_slices(([[turn.inputs] for turn in turns], [[turn.ewp, turn.player, turn.move] for turn in turns])).batch(32)
    for setup, scores in dataset:
      with tf.GradientTape() as tape:
        predictions = player(setup, training=True)
        mps, sps = predictions
        loss = []
        for mp, sp, [ewp, pi, mi] in zip(mps, sps, scores):
          prob = mp[int(mi)]
          bwp = sp[int(pi)]
          loss.append(- prob * (ewp - bwp) * (1 + (ewp - bwp)))
      gradients = tape.gradient(loss, player.trainable_variables)
      OPTIMIZER.apply_gradients(zip(gradients, player.trainable_variables))



