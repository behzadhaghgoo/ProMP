import argparse
import datetime
import os
import json

import dateutil.tz
import numpy as np

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.multitask_env import MultiClassMultiTaskEnv
from maml_zoo.envs.rl2_env import rl2env
from maml_zoo.algos.ppo import PPO
from maml_zoo.trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.rl2_sample_processor import RL2SampleProcessor
from maml_zoo.policies.gaussian_rnn_policy import GaussianRNNPolicy
from maml_zoo.logger import logger



maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


def main(config):
    from env_list import EASY_MODE_DICT, EASY_MODE_ARGS_KWARGS

    baseline = LinearFeatureBaseline()
    env = rl2env(MultiClassMultiTaskEnv(
        task_env_cls_dict=EASY_MODE_DICT,
        task_args_kwargs=EASY_MODE_ARGS_KWARGS))

    obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1

    policy = GaussianRNNPolicy(
        name="meta-policy",
        obs_dim=obs_dim,
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=config['meta_batch_size'],
        hidden_sizes=config['hidden_sizes'],
        cell_type=config['cell_type']
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('variant_index', metavar='variant_index', type=int,
                    help='The index of variants to use for experiment')
    args = parser.parse_args()

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    idx = args.variant_index
    logger.configure(dir='./data/rl2/test_{}_{}'.format(idx, timestamp), format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'],
                     snapshot_mode='gap', snapshot_gap=5,)
    config = json.load(open("./corl/rl2/configs/easy_mode_config{}.json".format(idx), 'r'))
    json.dump(config, open(maml_zoo_path + '/data/rl2/test_{}_{}/params.json'.format(idx, timestamp), 'w'))
    main(config)
