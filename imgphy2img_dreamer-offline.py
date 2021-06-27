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
  config.steps = 2e5
  config.eval_every = 5000
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  config.port = 8080
  # Environment.
  config.task = 'dmc_walker_walk'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 1
  config.time_limit = 500
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
  config.kl_scale = 1.0
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
    phy_post, phy_prior = self._phy_dynamics.observe(data['input_phy'], data['action'])
    # Get features
    phy_feat = self._phy_dynamics.get_feat(phy_post)
    # Reconstruct
    physics_pred = self._physics(phy_feat)
    # Observation
    embed = self._encode(data)
    env_post, env_prior = self._env_dynamics.observe(embed, data['prev_phy'])
    env_feat = self._env_dynamics.get_feat(env_post)
    env_pred = self._decode(env_feat)
    # Get features
    img_feat = self._env_dynamics.get_feat(env_post)
    full_feat = tf.concat([img_feat, phy_feat],-1)
    with tf.GradientTape() as reward_tape:
        reward_pred = self._reward(tf.stop_gradient(full_feat))
        err_reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
        reward_loss = - err_reward
     
    # Actor
    with tf.GradientTape() as actor_tape:
      # Imagination
      img_env_feat, img_phy_feat, img_full_feat = self._imagine_ahead(env_post, phy_post)
      # Actor training
      reward = self._reward(img_full_feat).mode()
      if self._c.pcont:
        pcont = self._pcont(img_env_feat).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      value = self._value(img_full_feat).mode()
      returns = tools.lambda_return(
          reward[:-1], value[:-1], pcont[:-1],
          bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * returns)

    # Value
    with tf.GradientTape() as value_tape:
      value_pred = self._value(img_full_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)
    reward_norm = self._reward_opt(reward_tape, reward_loss)

    if self._c.log_scalars:
      self._scalar_summaries(
          data, full_feat, value_loss, actor_loss, value_norm,
          actor_norm, reward_loss, reward_norm)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._env_dynamics = models.RSSMv2(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._phy_dynamics = models.RSSMv2(self._c.phy_stoch_size, self._c.phy_deter_size, self._c.phy_deter_size)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    self._physics = models.DenseDecoder([3], 1, self._c.phy_num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._reward_opt = Optimizer('reward', [self._reward], self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def _imagine_ahead(self, env_post, phy_post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    # Get initial states
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    env_start = {k: flatten(v) for k, v in env_post.items()}
    phy_start = {k: flatten(v) for k, v in phy_post.items()}

    # Define Policy and Physics functions
    policy = lambda env_state, phy_state: self._actor(
        tf.concat([tf.stop_gradient(self._env_dynamics.get_feat(env_state)),
                         tf.stop_gradient(self._phy_dynamics.get_feat(phy_state))],-1)).sample()
    physics = lambda state: self._physics(self._phy_dynamics.get_feat(state)).mode()
    # Run imagination
    env_states, phy_states = tools.forward_sync_RSSMv2(
        self._env_dynamics.img_step,
        self._phy_dynamics.img_step,
        env_start, phy_start,
        policy, physics,
        tf.range(self._c.horizon))
    # Collect features
    img_env_feat = self._env_dynamics.get_feat(env_states)
    img_phy_feat = self._phy_dynamics.get_feat(phy_states)
    img_full_feat = tf.concat([img_env_feat, img_phy_feat],-1)
    return img_env_feat, img_phy_feat, img_full_feat

  def _scalar_summaries(self, data, feat, value_loss, actor_loss,
          value_norm, actor_norm, reward_loss, reward_norm):
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['reward_grad_norm'].update_state(reward_norm)
    self._metrics['reward_loss'].update_state(reward_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())

  @tf.function
  def image_summaries(self, data):
    # Real 
    env_truth = data['image'][:6] + 0.5
    # Initial states (5 steps warmup)
    embed = self._encode(data)
    env_init, _ = self._env_dynamics.observe(embed[:6, :5], data['prev_phy'][:6, :5])
    env_init_feat = self._env_dynamics.get_feat(env_init)
    env_init = {k: v[:, -1] for k, v in env_init.items()}
    # Environment imagination
    env_prior = self._env_dynamics.imagine(data['prev_phy'][:6, 5:], env_init) 
    env_feat = self._env_dynamics.get_feat(env_prior)
    # Environment reconstruction
    env_obs = self._decode(env_init_feat).mode()
    openl = self._decode(env_feat).mode()
    env_model = tf.concat([env_obs[:, :5] + 0.5, openl + 0.5], 1)
    error = (env_model - env_truth + 1) / 2
    openl = tf.concat([env_truth, env_model, error], 2)
    return openl

  @tf.function
  def plot_dynamics(self, data):
    # Real 
    phy_truth = data['physics'][:3]
    # Initial states (5 steps warmup)
    phy_init, _ = self._phy_dynamics.observe(data['input_phy'][:3, :5], data['action'][:3, :5])
    phy_init_feat = self._phy_dynamics.get_feat(phy_init)
    phy_init = {k: v[:, -1] for k, v in phy_init.items()}
    # Physics imagination
    phy_prior = self._phy_dynamics.imagine(data['action'][:3, 5:], phy_init, sample=False)
    phy_feat = self._phy_dynamics.get_feat(phy_prior)
    # Physics reconstruction
    phy_obs = self._physics(phy_init_feat).mode()
    phy_pred = self._physics(phy_feat).mode()
    # Uncertainty
    phy_obs_std = self._physics(phy_init_feat).stddev()
    phy_pred_std = self._physics(phy_feat).stddev()
    # Concat and dump
    phy_model = tf.concat([phy_obs, phy_pred], 1)
    phy_model_std = tf.concat([phy_obs_std, phy_pred_std], 1)
    return phy_model, phy_model_std, phy_truth

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
    #obs['input_phy'] = tf.cast(tf.expand_dims(obs['physics'],-1),dtype)
    obs['input_phy'] = tf.cast(obs['physics'],dtype)
    obs['prev_phy'] = tf.cast(obs['physics_d'],dtype)
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
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

def summarize_episode(config, datadir, writer, prefix, step):
  list_of_files = glob.glob(str(datadir)+'/*.npz')
  latest_file = max(list_of_files, key=os.path.getctime)
  episode = np.load(latest_file)
  episode = {k: episode[k] for k in episode.keys()}
  episodes, steps = tools.count_episodes(datadir)
  print(episodes, steps)
  length = (len(episode['reward']) - 1) * config.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'episodes', episodes)]
  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
    #if prefix == 'test':
    tools.video_summary(f'sim/{prefix}/video', episode['image'][None], step=step)

def summarize_train(data, agent, step):
  with agent._writer.as_default(): 
    tf.summary.experimental.set_step(step)
    tools.video_summary('agent/environment_reconstruction',np.array(agent.image_summaries(data)),step = step)
    rec_phy, rec_phy_std, true_phy = agent.plot_dynamics(data)
    tools.plot_summary('agent/dynamics_reconstruction', np.array(rec_phy), np.array(rec_phy_std), np.array(true_phy), step=step)

def get_last_episode_reward(config, datadir, writer):
  list_of_files = glob.glob(str(datadir)+'/*.npz')
  latest_file = max(list_of_files, key=os.path.getctime)
  episode = np.load(latest_file)
  episode = {k: episode[k] for k in episode.keys()}
  ret = float(episode['reward'][-int(len(episode['reward'])/2):].sum())
  return ret

def make_env(config, writer, prefix, datadir, store):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, (64, 64), grayscale=False,
        life_done=True, sticky_actions=True)
    env = wrappers.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(
      lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
  env = wrappers.Collect(env, callbacks, config.precision)
  env = wrappers.RewardObs(env)
  return env


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

  has_trained = False
  
  step = count_steps(datadir, config)
  # Warm Up
  if step < 20000:
    raise ValueError('Dataset is too small')

  agent = Dreamer(config, datadir, actspace, writer)
  #if (config.logdir / 'variables.pkl').exists():
  #  print('Loading checkpoint.')
  #  agent.load(config.logdir / 'variables.pkl')

  print('Loading physics_dynamics.')
  agent._phy_dynamics.load(config.logdir / 'phy_dynamics_weights.pkl')
  print('Loading physics_reconstruction')
  agent._physics.load(config.logdir / 'physics_weights.pkl')
  print('Loading high-dim encoder')
  agent._encode.load(os.path.join(config.logdir,'encoder_weights.pkl'))
  print('Loading high-dim decoder')
  agent._decode.load(os.path.join(config.logdir,'decoder_weights.pkl'))
  print('Loading env dynamics')
  agent._env_dynamics.load(os.path.join(config.logdir,'env_dynamics_weights.pkl'))
  print('Loading reward')
  agent._reward.load(os.path.join(config.logdir,'reward_weights.pkl'))
  step = 0
  # Train and Evaluate continously
  while step < config.steps:
    plt.close('all')
    # Train
    print('Training for 100 steps')
    agent._step.assign(step)
    step = agent._step.numpy().item()
    tf.summary.experimental.set_step(step)

    log = agent._should_log(step)
    for train_step in range(100):
      data = next(agent._dataset)
      agent.train(data)
    step += 100
    if log:  
      summarize_train(data, agent, step)
      agent._write_summaries()
    has_trained=True

    # Save model after each training so ROS can load the new one.
    agent.save(config.logdir / 'variables.pkl')
    agent._actor.save(os.path.join(config.logdir,'actor_weights.pkl'))



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
