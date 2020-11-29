import tensorflow as tf

class Player(tf.keras.Model):
  def __init__(self, game):
    super(Player, self).__init__()
    self.input_layer = tf.keras.layers.Flatten(input_shape=(game.num_inputs,))
    self.hidden_layer = [tf.keras.layers.Dense(256, activation='relu') for x in range(2)]
    self.dropout_layer = tf.keras.layers.Dropout(0.2)
    self.actor_layer = tf.keras.layers.Dense(game.num_actions, activation='softmax')
    self.critic_layer = tf.keras.layers.Dense(game.num_players, activation='softmax')
  def call(self, x):
    x = self.input_layer(x)
    for layer in self.hidden_layer:
      x = layer(x)
    x = self.dropout_layer(x)
    a = self.actor_layer(x)
    c = self.critic_layer(x)
    return (a, c)
    
