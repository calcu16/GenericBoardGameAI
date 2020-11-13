import tensorflow as tf

class Turn:
  def __init__(self, inputs, player, move, scores):
    self.inputs = inputs
    self.player = player
    self.move = move
    self.bwp = scores[player]
    self.ewp = None
  def updateWinner(self, winners):
    self.ewp = winners[self.player]
  def updateScore(self, scores):
    self.ewp = scores[self.player]
  def deltaScore(self):
    return self.ewp - self.bwp
  def __repr__(self):
    return "[Turn: player: %s, move: %s, bwp: %0.2f, ewp: %0.2f]" % (self.player, self.move, self.bwp, self.ewp)

class Game:
  def __init__(self, game):
    self.game = game.clone()
    self.pi = None
    self.lturn = None
    self.nturn = 0
  def activePlayer(self):
    self.pi = self.game.activePlayer()
    return self.pi is not None
  def move(self, inp, rng_prob, mp, scores):
    self.nturn += 1
    if self.lturn is not None:
      self.lturn.updateScore(scores)
    if tf.random.uniform(shape=[1])[0].numpy() < rng_prob:
      mp = [1.0 for p in mp]
    m = tf.random.categorical([mp], 1)[0, 0].numpy()
    if m >= 9:
      print(mp)
    self.lturn = Turn(inp, self.pi, m, scores)
    self.game.move(m)
    return self.lturn
  def updateWinner(self):
    self.lturn.updateWinner(self.game.winner())

OPTIMIZER = tf.keras.optimizers.Adam()
def train(game, player, rng_prob = 0.0, num_games = 256, num_training_epochs = 5, debug = False):
  global OPTIMIZER
  players = [player for i in range(game.num_players)]
  pturns = [[]] * len(players)
  gs = [Game(game) for i in range(num_games)]
  dgs = []
  while gs:
    pgs = [(player, []) for player in players]
    for g in gs:
      if g.activePlayer():
        pgs[g.pi][1].append(g)
      else:
        g.updateWinner()
        dgs.append(g)
    for (p, gs) in pgs:
      if not gs: continue
      inp = [g.game.inputs() for g in gs]
      results = player.predict(inp)
      for g, inp, (mp, scores) in zip(gs, inp, zip(*results)):
        pturns[g.pi].append(g.move(inp, rng_prob,  mp, scores))

    gs = [g for p,gs in pgs for g in gs]
  if debug:
    print(pturns)
  print("Average game length: " + str(sum(g.nturn for g in dgs) / len(dgs)))

  critic_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
  def actor_loss_fn(mp, mi, ewp, bwp):
    m = mp[int(mi)]
    loss = -m * (ewp - bwp)
    if ewp - bwp < 0.1:
      loss += (max(mp) - m) / 10.0
    return loss
  for i in range(num_training_epochs):
    for turns in pturns:
      dataset = tf.data.Dataset.from_tensor_slices(([[turn.inputs] for turn in turns], [[turn.ewp, turn.bwp, turn.player, turn.move] for turn in turns])).batch(32)
      for setup, scores in dataset:
        with tf.GradientTape() as tape:
          predictions = player(setup, training=True)
          if debug:
            print(predictions)
          mps, sps = predictions
          critic_true, critic_pred, actor_losses = zip(*[(max(ewp, 0.0), sp[int(pi)], actor_loss_fn(mp, mi, ewp, bwp)) for mp, sp, [ewp, bwp, pi, mi] in zip(mps, sps, scores)])
          actor_loss = tf.math.reduce_sum(actor_losses)
          if debug:
            print("Critic true: " + str(critic_true))
            print("Critic pred: " + str(critic_pred))
          critic_loss = critic_loss_fn(critic_true, critic_pred)
          loss = actor_loss + critic_loss
        if debug:
          print("Actor: " + str(actor_loss))
          print("Critic: " + str(critic_loss))
        gradients = tape.gradient(loss, player.trainable_variables)
        OPTIMIZER.apply_gradients(zip(gradients, player.trainable_variables))
