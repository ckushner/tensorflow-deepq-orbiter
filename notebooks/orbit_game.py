from __future__ import print_function

# # coding: utf-8
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
# get_ipython().magic(u'matplotlib inline')

# porque?
# tf.

import numpy as np
import tempfile
import tensorflow as tf

from tf_rl.controller import DiscreteDeepQ, HumanController
from tf_rl.simulation import OrbiterGame
from tf_rl import simulate
from tf_rl.models import MLP


LOG_DIR = tempfile.mkdtemp()
print(LOG_DIR)


current_settings = {
    'G': 6.67e-11,

    'objects': [
        'planet',
        'asteroid',
    ],

    'colors': {
        'craft':    'yellow',
        'planet':   'green',
        'asteroid': 'red',
    },

    'object_reward': {
        'asteroid': -0.1,
    },

    'world_size': (1e7,1e7), # 10000km^2
    'image_size': 700,

    # DRAGON!
    'craft_initial_position': [9e6,9e6],
    'craft_initial_speed':    [0,   0],
    'craft_mass':             6000,
    'craft_radius':           2,

    'craft_rotations':        [-1, 0, 1],
    'craft_thrust_angle':     180,
    'craft_min_thrust':       0,    
    'craft_max_thrust':       400, # N
    'craft_step_thrust':      50,

    'planet_initial_position': [5e6,5e6],
    'planet_initial_speed':    [0,   0],
    'planet_mass':   5.9e24,
    'planet_radius': 6e6,

    # need/update?
    "maximum_speed": [1.5e4, 1.5e4],
    "asteroid_radius": 500.0,
    "asteroid_mass":   1e12,

    "num_objects": {
        "asteroid" : 0,
    },

    "num_observation_lines" : 32,
    "observation_line_length": 120.,
    "delta_v": 50
}


# create the game simulator
g = OrbiterGame(current_settings)


# Tensorflow business - it is always good to reset a graph before creating a new controller.
tf.ops.reset_default_graph()
session = tf.InteractiveSession()

# This little guy will let us run tensorboard
#      tensorboard --logdir [LOG_DIR]
journalist = tf.train.SummaryWriter(LOG_DIR)

# Brain maps from observation to Q values for different actions.
# Here it is a done using a multi layer perceptron with 2 hidden
# layers
brain = MLP([g.observation_size,], [100, 100, g.num_actions], 
            [tf.tanh, tf.tanh, tf.identity])

# The optimizer to use. Here we use RMSProp as recommended
# by the publication
optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.0001, decay=0.9)

# DiscreteDeepQ object
current_controller = DiscreteDeepQ(g.observation_size, g.num_actions, brain, optimizer, session,
                                   discount_rate=0.9, exploration_period=5000, max_experience=10000, 
                                   store_every_nth=4, train_every_nth=4,
                                   summary_writer=journalist)

session.run(tf.initialize_all_variables())
session.run(current_controller.target_network_update)
# graph was not available when journalist was created  
journalist.add_graph(session.graph_def)


fast_mode = False
if fast_mode:
    FPS, SPEED, RES = 1, 4.5, 0.1
else:
    FPS, SPEED, RES = 30, 1., 0.03

try:
    with tf.device("/cpu:0"): # can GPU?
        simulate(g, current_controller, fps = FPS,
                 simulation_resultion=RES,
                 actions_per_simulation_second=10,
                 speed=SPEED)
except KeyboardInterrupt:
    print("Interrupted")


session.run(current_controller.target_network_update)


current_controller.q_network.input_layer.Ws[0].eval()


current_controller.target_q_network.input_layer.Ws[0].eval()


# # Average Reward over time

g.plot_reward(smoothing=100)


# # Visualizing what the agent is seeing
# 
# Starting with the ray pointing all the way right, we have one row per ray in clockwise order.
# The numbers for each ray are the following:
# - first three numbers are normalized distances to the closest visible (intersecting with the ray) object. If no object is visible then all of them are $1$. If there's many objects in sight, then only the closest one is visible. The numbers represent distance to friend, enemy and wall in order.
# - the last two numbers represent the speed of moving object (x and y components). Speed of wall is ... zero.
# 
# Finally the last two numbers in the representation correspond to speed of the hero.

# In[8]:

g.__class__ = OrbiterGame
np.set_printoptions(formatter={'float': (lambda x: '%.2f' % (x,))})
x = g.observe()
new_shape = (x[:-2].shape[0]//g.eye_observation_size, g.eye_observation_size)
print(x[:-2].reshape(new_shape))
print(x[-2:])
g.to_html()


# In[ ]:



