import argparse
import os

import joblib
import json
import numpy as np
import tensorflow as tf

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.multitask_env import MultiClassMultiTaskEnv
from maml_zoo.logger import logger
from maml_zoo.meta_algos.trpo_maml import TRPOMAML
from maml_zoo.meta_trainer import Trainer
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor


maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


def main(config):

    from env_list import EASY_MODE_DICT, EASY_MODE_ARGS_KWARGS

    baseline = LinearFeatureBaseline()
    env = MultiClassMultiTaskEnv(
        task_env_cls_dict=EASY_MODE_DICT,
        task_args_kwargs=EASY_MODE_ARGS_KWARGS)

    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
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
    )
    trainer.train()


def resume(experiment, config, sess, start_itr):

    from env_list import EASY_MODE_DICT, EASY_MODE_ARGS_KWARGS

    baseline = LinearFeatureBaseline()
    env = MultiClassMultiTaskEnv(
        task_env_cls_dict=EASY_MODE_DICT,
        task_args_kwargs=EASY_MODE_ARGS_KWARGS)

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

    rand_num = np.random.uniform()
    idx = args.variant_index
    pkl = args.pkl
    config = args.config
    itr = args.itr

    if not config:
        config = json.load(open("./corl/configs/easy_mode_config{}.json".format(idx), 'r'))
    
    if pkl:
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir=maml_zoo_path + '/data/trpo/test_{}_{}'.format(idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='gap', snapshot_gap=5,)
            config = json.load(open(config, 'r'))
            json.dump(config, open(maml_zoo_path + '/data/trpo/test_{}_{}/params.json'.format(idx, rand_num), 'w'))
            resume(experiment, config, sess, itr)
    else:
        logger.configure(dir=maml_zoo_path + '/data/trpo/test_{}_{}'.format(idx, rand_num), format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='gap', snapshot_gap=5,)
        config = json.load(open("./corl/configs/easy_mode_config{}.json".format(idx), 'r'))
        json.dump(config, open(maml_zoo_path + '/data/trpo/test_{}_{}/params.json'.format(idx, rand_num), 'w'))
        main(config)
