import argparse
import collections
import functools
import json
import os
import glob
import pathlib
import sys
import time
import http.client
from bottle import Bottle, request
from matplotlib import pyplot as plt
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

def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.')
  config.seed = 0
  config.steps = 5e6
  config.eval_every = 5000
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  # Environment.
  config.task = 'dmc_walker_walk'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 1
  config.time_limit = 1000
  config.prefill = 5000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  # Model.
  config.deter_size = 300
  config.stoch_size = 30
  config.num_units = 400
  config.phy_deter_size = 50
  config.phy_stoch_size = 5
  config.phy_num_units = 60
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 0.5
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.97
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  return config


class Dreamer(tools.Module):

  def __init__(self, config, datadir, actspace, writer):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
 
    self._dataset = iter(load_dataset(datadir, self._c))
    self._build_model()

  def load(self, filename):
    super().load(filename)
    self._should_pretrain()

  @tf.function()
  def train(self, data, log_images=False):
    self._train(data, log_images)

  def _train(self, data, log_images):
    # World Model
    
    likes = tools.AttrDict()
    with tf.GradientTape() as phy_tape:
      # Observation
      phy_post, phy_prior = self._phy_dynamics.observe(data['input_phy'], data['action'])
      # Get features
      phy_feat = self._phy_dynamics.get_feat(phy_post)
      # Reconstruct
      physics_pred = self._physics(phy_feat)
      # Reconstruction errors
      likes.physics = tf.reduce_mean(physics_pred.log_prob(data['input_phy']))
      # Maximize the use of the internal state
      phy_prior_dist = self._phy_dynamics.get_dist(phy_prior)
      phy_post_dist = self._phy_dynamics.get_dist(phy_post)
      phy_div = tf.reduce_mean(tfd.kl_divergence(phy_post_dist, phy_prior_dist))
      phy_div = tf.maximum(phy_div, self._c.free_nats)
      # World model loss
      phy_loss = - likes.physics + self._c.kl_scale * phy_div
    
    phy_norm = self._phy_opt(phy_tape, phy_loss)

    if self._c.log_scalars:
      self._scalar_summaries(
          phy_prior_dist, phy_post_dist, likes, phy_div,
          phy_loss, phy_norm)
    if tf.equal(log_images, True):
      self._image_summaries(data, physics_pred)


  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    act = acts[self._c.dense_act]
    self._phy_dynamics = models.RSSMv2(self._c.phy_stoch_size, self._c.phy_deter_size, self._c.phy_deter_size)
    self._physics = models.DenseDecoder([3], 1, self._c.phy_num_units, act=act)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._phy_opt = Optimizer('physics_model', [self._physics, self._phy_dynamics], self._c.model_lr)
    self.train(next(self._dataset))

  def _scalar_summaries(
      self, prior_dist, post_dist, likes, div,
      phy_loss, phy_norm):
    self._metrics['phy_grad_norm'].update_state(phy_norm)
    self._metrics['phy_prior_ent'].update_state(prior_dist.entropy())
    self._metrics['phy_post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['phy_div'].update_state(div)
    self._metrics['phy_loss'].update_state(phy_loss)

  def _image_summaries(self, data, phy_pred):
    # Real 
    phy_truth = data['physics'][:6]
    # Reconstructions
    phy_rec = phy_pred.mode()[:6]
    # Initial states (5 steps warmup)
    phy_init, _ = self._phy_dynamics.observe(data['input_phy'][:6, :5], data['action'][:6, :5])
    phy_init = {k: v[:, -1] for k, v in phy_init.items()}
    # Physics imagination
    phy_prior = self._phy_dynamics.imagine(data['action'][:6, 5:], phy_init,
            sample=True)
    phy_feat = self._phy_dynamics.get_feat(phy_prior)
    # Physics reconstruction
    phy_pred = self._physics(phy_feat).mode()
    phy_model = tf.concat([phy_rec[:, :5], phy_pred], 1)
    phy = tf.concat([phy_truth, phy_model], 1)
    tools.graph_summary(self._writer, tools.plot_summary, 'agent/physics_reconstruction', phy)
    # Initial states (5 steps warmup)
    phy_init, _ = self._phy_dynamics.observe(data['input_phy'][:6, :5], data['action'][:6, :5])
    phy_init = {k: v[:, -1] for k, v in phy_init.items()}
    # Physics imagination no sampling
    phy_prior = self._phy_dynamics.imagine(data['action'][:6, 5:], phy_init,
            sample=False)
    phy_feat = self._phy_dynamics.get_feat(phy_prior)
    # Physics reconstruction
    phy_pred = self._physics(phy_feat).mode()
    phy_model = tf.concat([phy_rec[:, :5], phy_pred], 1)
    phy = tf.concat([phy_truth, phy_model], 1)
    tools.graph_summary(self._writer, tools.plot_summary, 'agent/physics_reconstruction_ns', phy)

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    self._writer.flush()

def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['input_phy'] = tf.cast(obs['physics'],dtype)
  return obs

def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat

def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset

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

  # Setting up Dreamer
  datadir = config.logdir / 'episodes'
  testdir = config.logdir / 'test_episodes'
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  actspace = gym.spaces.Box(np.array([-1,-1]),np.array([1,1]))
  writer.flush()

  agent = Dreamer(config, datadir, actspace, writer)
  if (config.logdir / 'phy_dynamics_weights.pkl').exists():
    print('Loading physics_dynamics.')
    _phy_dynamics.load(config.logdir / 'phy_dynamics_weights.pkl')
  if (config.logdir / 'physics_weights.pkl').exists():
    print('Loading physics_reconstruction')
    _phy_dynamics.load(config.logdir / 'physics_weights.pkl')

  # Train and Evaluate continously
  step = 0
  while step < config.steps:
    plt.close('all')
    print('Training for 100 steps')
    agent._step.assign(step)
    step = agent._step.numpy().item()
    tf.summary.experimental.set_step(step)
    log = agent._should_log(step)
    for train_step in range(100):
      log_images = agent._c.log_images and log and (train_step == 0)
      agent.train(next(agent._dataset), log_images)
    if log:
      agent._write_summaries()
    step += 100

    agent._phy_dynamics.save(os.path.join(config.logdir,'phy_dynamics_weights.pkl'))
    agent._physics.save(os.path.join(config.logdir,'physics_weights.pkl'))

if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())
