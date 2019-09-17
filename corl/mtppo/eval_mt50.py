import argparse
import os

import joblib
import json
import numpy as np
import tensorflow as tf

from maml_zoo.algos.ppo import PPO
from maml_zoo.samplers.single_sample_processor import SingleSampleProcessor

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.logger import logger
from maml_zoo.meta_algos.trpo_maml import TRPOMAML
from maml_zoo.trainer import Trainer
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor

from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv


maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


N_TASKS = 50
TASKNAME = 'mt50'


def mt_test(experiment, config, sess, start_itr, pkl, algo_name):

    config['rollouts_per_meta_task'] = 10
    config['max_path_length'] = 150

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

    config['meta_batch_size'] = 50

    baseline = LinearFeatureBaseline()
    policy = experiment['policy']

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
        name='mt50_{}'.format(algo_name),
    )

    trainer.train(test_time=True)


def mt50_test_batch(folder, config, start_itr, algo_name, start, end):
    config['rollouts_per_meta_task'] = 10
    config['max_path_length'] = 150

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
    config['meta_batch_size'] = 50

    baseline = LinearFeatureBaseline()
    sample_processor = SingleSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    sampler = MAMLSampler(
        env=env,
        policy=None,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        meta_batch_size=len(MT50_CLS_DICT.keys()),
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=config['envs_per_task'],
    )

    from os import listdir
    all_pkls = ['itr_{}.pkl'.format(i) for i in range(start, end)]

    for p in all_pkls:
        full_path = os.path.join(folder, p)
        eval_single(env, full_path, sampler, sample_processor, config, algo_name, full_path)


def eval_single(env, pkl_file_path, sampler, sample_processor, config, algo_name, full_path):
    """
    load policy-> replace sampler's policy-> rebuild tf graph wit new session-> eval
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with open(pkl_file_path, 'rb') as f:
                experiment = joblib.load(f)
                policy = experiment['policy']
                sampler.policy = policy

                algo = PPO(
                    policy=policy,
                    learning_rate=config['learning_rate']
                )
                trainer = Trainer(
                    algo=algo,
                    env=env,
                    sampler=sampler,
                    sample_processor=sample_processor,
                    policy=policy,
                    n_itr=config['n_itr'],
                    name='mt50_{}'.format(algo_name),
                    pkl=full_path,
                )

                trainer.train(test_time=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('variant_index', metavar='variant_index', type=int,
                    help='The index of variants to use for experiment')
    parser.add_argument('--dir', metavar='dir', type=str,
                    help='The path of the folder that contains pkl files',
                    default=None, required=False)
    parser.add_argument('--pkl', metavar='pkl', type=str,
                    help='The path of the pkl file', 
                    default=None, required=False)
    parser.add_argument('--algo', metavar='algo', type=str,
                    help='Algo name', 
                    default=None, required=False)
    parser.add_argument('--itr', metavar='itr', type=int,
                    help='The start itr of the resuming experiment', 
                    default=0, required=False)

    parser.add_argument('--start', metavar='start', type=int,
                    default=0, required=False)
    parser.add_argument('--end', metavar='end', type=int,
                    default=0, required=False)
    args = parser.parse_args()

    rand_num = np.random.uniform()
    idx = args.variant_index
    pkl = args.pkl
    folder = args.dir
    algo = args.algo
    itr = args.itr

    start = args.start
    end = args.end
    if pkl:
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir=maml_zoo_path + '/data/mtppo_test/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all',)
            config = json.load(open('./corl/mtppo/mt50_config.json', 'r'))
            json.dump(config, open(maml_zoo_path + '/data/mtppo_test/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
            mt_test(experiment, config, sess, itr, pkl, algo)
    elif folder:
        logger.configure(dir=maml_zoo_path + '/data/mtppo_test/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
            snapshot_mode='all',)
        config = json.load(open('./corl/mtppo/mt50_config.json', 'r'))
        json.dump(config, open(maml_zoo_path + '/data/mtppo_test/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
        mt50_test_batch(folder, config, itr, algo, start, end)
    else:
        print('Please provide a pkl file')
