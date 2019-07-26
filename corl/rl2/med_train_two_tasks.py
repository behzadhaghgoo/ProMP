import argparse
import datetime
import os
import json

import dateutil.tz
import joblib
import numpy as np
import tensorflow as tf

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.multitask_env import MultiClassMultiTaskEnv
from maml_zoo.envs.rl2_env import rl2env
from maml_zoo.algos.ppo import PPO
from maml_zoo.trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.rl2_sample_processor import RL2SampleProcessor
from maml_zoo.policies.gaussian_rnn_policy import GaussianRNNPolicy
from maml_zoo.logger import logger



def rl2_eval(experiment, config, sess, start_itr, all_params):
    import collections
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv

    TRAIN_DICT = collections.OrderedDict([
        ('reach', SawyerReachPushPickPlace6DOFEnv),
        ('dial_turn', SawyerDialTurn6DOFEnv),
    ])

    TRAIN_ARGS_KWARGS = {
        'reach': {
            "args": [],
            "kwargs": {
                'tasks': [{'goal': np.array([-0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}],
                'multitask': False,
                'obs_type': 'plain',
                'if_render': False,
                'random_init': True,
            }
        },
        'dial_turn': {
            "args": [],
            "kwargs": {
                'tasks': [{'goal': np.array([0., 0.73, 0.08]), 'obj_init_pos':np.array([0, 0.7, 0.05])}],
                'multitask': False,
                'obs_type': 'plain',
                'if_render': False,
                'random_init': True,
            }
        },
    }

    baseline = LinearFeatureBaseline()
    env = rl2env(MultiClassMultiTaskEnv(
        task_env_cls_dict=TRAIN_DICT,
        task_args_kwargs=TRAIN_ARGS_KWARGS))

    obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1

    policy = GaussianRNNPolicy(
        name="meta-policy",
        obs_dim=obs_dim,
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=config['meta_batch_size'],
        hidden_sizes=config['hidden_sizes'],
        cell_type=config['cell_type'],
        init_std=2.
    )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=config['envs_per_task']
    )

    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
    )
    trainer.train()


def resume(experiment, config, sess, start_itr):

    from medium_env_list import TRAIN_DICT, TRAIN_ARGS_KWARGS

    baseline = LinearFeatureBaseline()
    env = rl2env(MultiClassMultiTaskEnv(
        task_env_cls_dict=TRAIN_DICT,
        task_args_kwargs=TRAIN_ARGS_KWARGS))

    policy = experiment['policy']

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=config['envs_per_task']
    )

    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        sess=sess,
        start_itr=start_itr,
    )

    trainer.train()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('variant_index', metavar='variant_index', type=int,
                    help='The index of variants to use for experiment')
    parser.add_argument('--pkl', metavar='pkl', type=str,
                    help='The path of the pkl file',
                    default=None, required=False)
    parser.add_argument('--config', metavar='config', type=str,
                    help='The path to the config file',
                    default=None, required=False)
    parser.add_argument('--itr', metavar='itr', type=int,
                    help='The start itr of the resuming experiment',
                    default=0, required=False)
    args = parser.parse_args()

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    idx = args.variant_index
    pkl = args.pkl
    config = args.config
    itr = args.itr

    if not config:
        config = json.load(open("./corl/rl2/configs/medium_mode_config{}.json".format(idx), 'r'))

    if pkl:
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir='./data/rl2/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'],
                     snapshot_mode='gap', snapshot_gap=5,)
            config = json.load(open(config, 'r'))
            json.dump(config, open('./data/rl2/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
            resume(experiment, config, sess, itr)
    else:
        logger.configure(dir='./data/rl2/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'],
                     snapshot_mode='gap', snapshot_gap=5,)
        config = json.load(open("./corl/rl2/configs/medium_mode_config{}.json".format(idx), 'r'))
        json.dump(config, open('./data/rl2/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
        main(config)