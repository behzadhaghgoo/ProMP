import argparse
import datetime
import os

import dateutil.tz
import joblib
import json
import gym
import numpy as np
import tensorflow as tf

from maml_zoo.logger import logger
from maml_zoo.algos.trpo import TRPO
from maml_zoo.baselines import LinearFeatureBaseline
from maml_zoo.policies import GaussianMLPPolicy
from maml_zoo.samplers import MAMLSampler
from maml_zoo.samplers.single_sample_processor import SingleSampleProcessor
from maml_zoo.trainer import Trainer

from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv


maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])
N_TASKS = 50

def run_experiment(**kwargs):

    from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
    MT50_CLS_DICT = {}
    MT50_ARGS_KWARGS = {}
    for k in HARD_MODE_CLS_DICT.keys():
        for i in HARD_MODE_CLS_DICT[k].keys():
            key = '{}-{}'.format(k, i)
            MT50_CLS_DICT[key] = HARD_MODE_CLS_DICT[k][i]
            MT50_ARGS_KWARGS[key] = HARD_MODE_ARGS_KWARGS[k][i]
    assert len(MT50_CLS_DICT.keys()) == N_TASKS

    mt50 = MultiClassMultiTaskEnv(
            task_env_cls_dict=MT50_CLS_DICT,
            task_args_kwargs=MT50_ARGS_KWARGS,
            sample_goals=False,
            obs_type='with_goal_idx',
            sample_all=True,)
    # discretize goal space
    goals_dict = {
        t: [e.goal.copy()]
        for t, e in zip(mt50._task_names, mt50._task_envs)}
    mt50.discretize_goal_space(goals_dict)
    # reach, push, pickplace are different
    mt50._task_envs[0].obs_type = 'reach'
    mt50._task_envs[1].obs_type = 'pickplace'
    mt50._task_envs[2].obs_type = 'push'
    mt50._task_envs[3].obs_type = 'reach'
    mt50._task_envs[4].obs_type = 'pickplace'
    mt50._task_envs[5].obs_type = 'push'
    mt50._task_envs[0].goal = np.array([-0.1, 0.8, 0.2])
    mt50._task_envs[1].goal = np.array([0.1, 0.8, 0.2])
    mt50._task_envs[2].goal = np.array([0.1, 0.8, 0.02])
    mt50._task_envs[3].goal = np.array([-0.05, 0.8, 0.2])
    mt50._task_envs[4].goal = np.array([0.05, 0.8, 0.2])
    mt50._task_envs[5].goal = np.array([0.05, 0.8, 0.015])

    env = mt50

    # dimensions
    obs_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.observation_space.shape)

    # baseline
    baseline = LinearFeatureBaseline()

    # policy
    policy = GaussianMLPPolicy(
        name='policy',
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=kwargs['hidden_sizes'],
        learn_std=kwargs['learn_std'],
        init_std=kwargs['init_std'],
    )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=kwargs['rollouts_per_meta_task'],
        meta_batch_size=len(MT50_CLS_DICT.keys()),
        max_path_length=kwargs['max_path_length'],
        parallel=kwargs['parallel'],
        envs_per_task=kwargs['envs_per_task'],
    )

    sample_processor = SingleSampleProcessor(
        baseline=baseline,
        discount=kwargs['discount'],
        gae_lambda=kwargs['gae_lambda'],
        normalize_adv=kwargs['normalize_adv'],
        positive_adv=kwargs['positive_adv'],
    )

    algo = TRPO(
        policy=policy,
        learning_rate=kwargs['learning_rate']
    )

    trainer = Trainer(
        algo=algo,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        policy=policy,
        n_itr=kwargs['n_itr'],
    )

    trainer.train()

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('variant_index', metavar='variant_index', type=int,
                    help='The index of variants to use for experiment')
    parser.add_argument('--pkl', metavar='pkl', type=str,
                    help='The path of the pkl file', 
                    default=None, required=False)

    parser.add_argument('--itr', metavar='itr', type=int,
                    help='The start itr of the resuming experiment', 
                    default=0, required=False)
    args = parser.parse_args()

    idx = args.variant_index
    pkl = args.pkl

    itr = args.itr
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    if pkl:
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir=maml_zoo_path + '/data/mtppo/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all')
            config = json.load(open("./corl/mtppo/mt50_config.json", 'r'))
            json.dump(config, open(maml_zoo_path + '/data/mtppo/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
            resume(experiment, config, sess, itr)
    else:
        logger.configure(dir=maml_zoo_path + '/data/mtppo/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all')
        config = json.load(open("./corl/mtppo/mt50_config.json", 'r'))
        json.dump(config, open(maml_zoo_path + '/data/mtppo/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
        run_experiment(**config)
