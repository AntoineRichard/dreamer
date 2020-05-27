import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import http.client
from bottle import Bottle, request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers
import simpler_dreamer

def main(config):
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  config.steps = int(config.steps)
  config.logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', config.logdir)

  # Create environments.
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  actspace = gym.spaces.Box(np.array([-1,-1]),np.array([1,1]))
  writer.flush()

  
  steps = simpler_dreamer.count_steps(datadir, config)
  #print(steps)
  if steps < 100:
    c = http.client.HTTPConnection('localhost', 8080)
    c.request('POST', '/toServer', '{"random": 1, "steps":500, "repeat":9, "discount":1.0, "training": 1}')
    doc = c.getresponse().read()
  step = simpler_dreamer.count_steps(datadir, config)
  agent = simpler_dreamer.Dreamer(config, datadir, actspace)
  if (config.logdir / 'variables.pkl').exists():
    print('Loading checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  keys = ['image','reward']
  training = True
  for i in range(1000):
      episode = np.load('/mnt/nvme-storage/antoine/DREAMER/dreamer/logdir/turtle_sim/dreamer/1/episodes/20200513T214109-7f63367b4d41460bb2efa78e30430ab9-502.npz')
      episode = {k: episode[k] for k in episode.keys()}
      state=None
      for i in range(490):
          obs = {k: [episode[k][i]] for k in keys}
          action, state = agent.policy(obs, state, training)

  while step < config.steps:
    print(simpler_dreamer.count_steps(datadir, config))
    print('Training for 100 steps')
    for train_step in range(100):
      log = agent._should_log(step)
      log_images = agent._c.log_images and log and train_step == 0
      #print(log_images)
      agent.train(next(agent._dataset), log_images)
    agent.save(config.logdir / 'variables.pkl')
    c = http.client.HTTPConnection('localhost', 8080)
    c.request('POST', '/toServer', '{"random": 0, "steps":500, "repeat":0, "discount":1.0, "training": 1}')
    doc = c.getresponse().read()
  #print(count_steps(datadir, config))
  #c = http.client.HTTPConnection('localhost', 8080)
  #c.request('POST', '/toServer', '{"random": 0, "steps":500, "repeat":0, "discount":1.0, "training": 1}')
  #doc = c.getresponse().read()
  #print(count_steps(datadir, config))




  #while step < config.steps:
  #  print('Start evaluation.')
  #  tools.simulate(
  #      functools.partial(agent, training=False), test_envs, episodes=1)
  #  writer.flush()
  #  print('Start collection.')
  #  steps = config.eval_every // config.action_repeat
  #  state = tools.simulate(agent, train_envs, steps, state=state)
  #  step = count_steps(datadir, config)
  #  agent.save(config.logdir / 'variables.pkl')
  #for env in train_envs:# + test_envs:
  #  env.close()


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in simpler_dreamer.define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())
