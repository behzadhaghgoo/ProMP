import argparse
import os
import dateutil.tz
import datetime

import joblib
import json
import numpy as np
import tensorflow as tf

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.logger import logger
from maml_zoo.meta_algos.trpo_maml import TRPOMAML
from maml_zoo.meta_trainer import Trainer
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor

from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv


maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])
TASKNAME = 'hard-test-testenvs'


def maml_test(experiment, config, sess, start_itr, pkl):

    config['meta_batch_size'] = 45  # Sampler force this to be the original meta_batch_size
    config['rollouts_per_meta_task'] = 10
    config['max_path_length'] = 150

    from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS

    test_cls = HARD_MODE_CLS_DICT['test']
    test_args = HARD_MODE_ARGS_KWARGS['test']
    test_cls[28] = HARD_MODE_CLS_DICT['train']['28']
    test_args[28] = HARD_MODE_ARGS_KWARGS['train']['28']


    baseline = LinearFeatureBaseline()
    # goals are sampled and set anyways so we don't care about the default goal of reach
    # pick_place, push are the same.
    env = MultiClassMultiTaskEnv(
        task_env_cls_dict=test_cls,
        task_args_kwargs=test_args,
        sample_goals=True,
        obs_type='plain',
        sample_all=True,
    )

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

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = TRPOMAML(
        policy=policy,
        step_size=config['step_size'],
        inner_type=config['inner_type'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        inner_lr=config['inner_lr']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],  # This is repeated in MAMLPPO, it's confusing
        sess=sess,
        start_itr=start_itr,
        pkl=pkl,
        name='testenvs',
    )

    trainer.train(test_time=True)


def maml_test_batch(folder, config, start_itr, start, end, suffix):

    config['rollouts_per_meta_task'] = 10
    config['max_path_length'] = 150

    from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS

    test_cls = HARD_MODE_CLS_DICT['test']
    test_args = HARD_MODE_ARGS_KWARGS['test']
    test_cls[28] = HARD_MODE_CLS_DICT['train']['28']
    test_args[28] = HARD_MODE_ARGS_KWARGS['train']['28']


    # goals are sampled and set anyways so we don't care about the default goal of reach
    # pick_place, push are the same.
    env = MultiClassMultiTaskEnv(
        task_env_cls_dict=test_cls,
        task_args_kwargs=test_args,
        sample_goals=True,
        obs_type='plain',
        sample_all=True,
    )
    config['meta_batch_size'] = len(HARD_MODE_CLS_DICT['train'].keys())

    baseline = LinearFeatureBaseline()
    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    sampler = MAMLSampler(
        env=env,
        policy=None,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=config['envs_per_task'],
    )

    all_pkls = ['itr_{}.pkl'.format(i) for i in range(start, end)]
    for p in all_pkls:
        full_path = os.path.join(folder, p)
        eval_single(env, full_path, sampler, sample_processor, config, full_path, suffix)


def eval_single(env, pkl_file_path, sampler, sample_processor, config, full_path, suffix):
    """
    load policy-> replace sampler's policy-> rebuild tf graph wit new session-> eval
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with open(pkl_file_path, 'rb') as f:
                experiment = joblib.load(f)
                policy = experiment['policy']
                sampler.policy = policy

                algo = TRPOMAML(
                    policy=policy,
                    step_size=config['step_size'],
                    inner_type=config['inner_type'],
                    meta_batch_size=config['meta_batch_size'],
                    num_inner_grad_steps=config['num_inner_grad_steps'],
                    inner_lr=config['inner_lr']
                )
                trainer = Trainer(
                    algo=algo,
                    policy=policy,
                    env=env,
                    sampler=sampler,
                    sample_processor=sample_processor,
                    n_itr=config['n_itr'],
                    num_inner_grad_steps=config['num_inner_grad_steps'],  # This is repeated in MAMLPPO, it's confusing
                    sess=sess,
                    start_itr=0,
                    pkl=full_path,
                    name='hard_testenvs_{}'.format(suffix),
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
    parser.add_argument('--config', metavar='config', type=str,
                    help='The path to the config file', 
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
    config_file = args.config
    itr = args.itr

    start = args.start
    end = args.end

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    from os import listdir
    from os.path import isfile
    import os.path
    pkls = [file for file in listdir(folder) if '.pkl' in file]

    if not config_file:
        config_file = '/root/code/ProMP/corl/configs/hard_mode_config{}.json'.format(idx)
    
    if pkl:
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir=maml_zoo_path + '/data/maml_test/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all',)
            config = json.load(open(config_file, 'r'))
            json.dump(config, open(maml_zoo_path + '/data/maml_test/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
            maml_test(experiment, config, sess, itr, pkl)
    elif folder:
        logger.configure(dir=maml_zoo_path + '/data/maml_test/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
            snapshot_mode='all',)
        config = json.load(open(config_file, 'r'))
        json.dump(config, open(maml_zoo_path + '/data/maml_test/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
        maml_test_batch(folder, config, itr, start, end, timestamp)
    else:
        print('Please provide a pkl file')
