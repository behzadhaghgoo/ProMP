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
from maml_zoo.algos.ppo import PPO
from maml_zoo.baselines import LinearFeatureBaseline
from maml_zoo.policies import GaussianMLPPolicy
from maml_zoo.samplers import MAMLSampler
from maml_zoo.samplers.single_sample_processor import SingleSampleProcessor
from maml_zoo.trainer import Trainer

from metaworld.envs.mujoco.sawyer_xyz import SawyerNutDisassemble6DOFEnv
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS


print(EASY_MODE_ARGS_KWARGS)
maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


def run_experiment(**kwargs):
    env = MultiClassMultiTaskEnv(
        task_env_cls_dict=EASY_MODE_CLS_DICT,
        task_args_kwargs=EASY_MODE_ARGS_KWARGS,
        sample_goals=False,
        obs_type='with_goal_idx',
    )
    goals_dict = {
        t: [e.goal.copy()]
        for t, e in zip(env._task_names, env._task_envs)
    }
    print(goals_dict)
    env.discretize_goal_space(goals_dict)
    print(kwargs)

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
        meta_batch_size=len(EASY_MODE_CLS_DICT.keys()),
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

    # TODO check if ryan is using PPO or TRPO
    algo = PPO(
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
            logger.configure(dir=maml_zoo_path + '/data/trpo/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all')
            config = json.load(open(config, 'r'))
            json.dump(config, open(maml_zoo_path + '/data/trpo/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
            resume(experiment, config, sess, itr)
    else:
        logger.configure(dir=maml_zoo_path + '/data/trpo/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all')
        config = json.load(open("./rl_launchers/single_config.json", 'r'))
        json.dump(config, open(maml_zoo_path + '/data/trpo/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
        run_experiment(**config)
