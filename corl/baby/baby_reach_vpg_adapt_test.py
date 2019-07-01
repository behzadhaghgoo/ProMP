from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.point_envs.point_env_2d import MetaPointEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.algos.vpg import VPG
from maml_zoo.meta_tester import Tester
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.samplers import SampleProcessor
from maml_zoo.policies.gaussian_mlp_policy import GaussianMLPPolicy
import os
from maml_zoo.logger import logger
import json
import numpy as np
import tensorflow as tf
import argparse
import joblib


from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from baby_wrapper import BabyModeWrapper

maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

N_TASKS = 1
TASKNAME = 'reach'


def resume(experiment, config, sess, start_itr=0):
    pickled_env = experiment['env'].env
    pickled_tasks = pickled_env.tasks

    goal_low = np.array((-0.1, 0.8, 0.05))
    goal_high = np.array((0.1, 0.9, 0.3))

    goals = np.random.uniform(low=goal_low, high=goal_high, size=(N_TASKS, len(goal_low))).tolist()
    print(goals)

    tasks =[
        {'goal': np.array(g), 'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}
        for i, g in enumerate(goals)
    ]

    # replace the first task
    print('picked_tasks[0]:')
    print(pickled_tasks[0])
    print('tasks[0]:')
    print(tasks[0])
    tasks = [pickled_tasks[0]]

    baseline = LinearFeatureBaseline()
    env = BabyModeWrapper(SawyerReachPushPickPlace6DOFEnv(
        random_init=False,
        multitask=False,
        obs_type='plain',
        if_render=False,
        tasks=tasks,
    ))

    policy = experiment['policy']
    # policy.switch_to_pre_update()

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=config['envs_per_task'],
    )

    sample_processor = SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = VPG(
        policy=policy,
        learning_rate=config['learning_rate'],
        inner_type=config['inner_type'],
    )

    tester = Tester(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        sess=sess,
        task=None,
    )

    tester.train()


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

    rand_num = np.random.uniform()
    idx = args.variant_index
    pkl = args.pkl
    config = args.config
    itr = args.itr

    if not config:
        config = './corl/configs/baby_mode_vpg_config{}.json'.format(idx)

    if pkl:
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir=maml_zoo_path + '/data/vpg/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all',)
            config = json.load(open(config, 'r'))
            json.dump(config, open(maml_zoo_path + '/data/vpg/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
            resume(experiment, config, sess, itr)
    else:
        logger.configure(dir=maml_zoo_path + '/data/vpg/test_{}_{}_{}'.format(TASKNAME, idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='all',)
        config = json.load(open("./corl/configs/baby_mode_config{}.json".format(idx), 'r'))
        json.dump(config, open(maml_zoo_path + '/data/vpg/test_{}_{}_{}/params.json'.format(TASKNAME, idx, rand_num), 'w'))
        # main(config)


    # idx = np.random.randint(0, 1000)
    # logger.configure(dir=maml_zoo_path + '/data/vpg/test_%d' % idx, format_strs=['stdout', 'log', 'csv'],
    #                  snapshot_mode='last_gap')
    # config = json.load(open(maml_zoo_path + "/configs/vpg_maml_config.json", 'r'))
    # json.dump(config, open(maml_zoo_path + '/data/vpg/test_%d/params.json' % idx, 'w'))
    # main(config)
