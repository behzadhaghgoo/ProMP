import argparse
import os

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

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from baby_wrapper import BabyModeWrapper

maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

N_TASKS = 10
TASKNAME = 'pick_place'


def maml_test(experiment, config, sess, start_itr, pkl):

    pickled_env = experiment['env'].env
    pickled_tasks = pickled_env.tasks

    goal_low = np.array((0.05, 0.8, 0.2))
    goal_high = np.array((0.15, 0.9, 0.2))

    goals = np.random.uniform(low=goal_low, high=goal_high, size=(N_TASKS, len(goal_low))).tolist()
    print(goals)

    tasks =[
        {'goal': np.array(g), 'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'pick_place'}
        for i, g in enumerate(goals)
    ]

    baseline = LinearFeatureBaseline()
    env = BabyModeWrapper(SawyerReachPushPickPlace6DOFEnv(
        random_init=False,
        multitask=False,
        obs_type='plain',
        if_render=False,
        tasks=tasks,
    ))

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
    args = parser.parse_args()

    rand_num = np.random.uniform()
    idx = args.variant_index
    pkl = args.pkl
    folder = args.dir
    config_file = args.config
    itr = args.itr

    from os import listdir
    from os.path import isfile
    import os.path
    pkls = [file for file in listdir(folder) if '.pkl' in file]

    if not config_file:
        config_file = '/root/code/ProMP/corl/configs/baby_mode_config{}.json'.format(idx)
    
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
        for p in pkls:
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    with open(os.path.join(folder, p), 'rb') as file:
                        experiment = joblib.load(file)
                    logger.configure(dir=maml_zoo_path + '/data/maml_test/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                             snapshot_mode='all',)
                    config = json.load(open(config_file, 'r'))
                    json.dump(config, open(maml_zoo_path + '/data/maml_test/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
                    maml_test(experiment, config, sess, itr, p)
            import gc
            gc.collect()
    else:
        print('Please provide a pkl file')
