import tensorflow as tf

class Player(tf.keras.Model):
  def __init__(self, game):
    super(Player, self).__init__()
    self.input_layer = tf.keras.layers.Flatten(input_shape=game.input_shape)
    self.hidden_layer = tf.keras.layers.Dense(128, activation='relu')
    self.dropout_layer = tf.keras.layers.Dropout(0.2)
    self.action_layer = tf.keras.layers.Dense(game.num_actions, activation='softmax')
    self.critic_layer = tf.keras.layers.Dense(game.num_players, activation='softmax')
  def call(self, x):
    x = self.input_layer(x)
    x = self.hidden_layer(x)
    x = self.dropout_layer(x)
    a = self.action_layer(x)
    c = self.critic_layer(x)
    return (a, c)
    
