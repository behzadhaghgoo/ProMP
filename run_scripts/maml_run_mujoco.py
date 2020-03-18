import argparse
import time
import json
import os
import numpy as np
import sys
sys.path.append('~/cml_winter/') 
# sys.path.append('~/cml_winter/meta_policy_search/')
from meta_policy_search.utils.utils import set_seed, ClassEncoder
from meta_policy_search.utils import logger
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.meta_trainer import KAML_Test_Trainer # changed from meta_trainer_giraffe 
from meta_policy_search.meta_algos.trpo_maml import TRPOMAML
from meta_policy_search.envs.normalized_env import normalize
# from meta_policy_search.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.swimmer_rand_vel import SwimmerRandVelEnv


from meta_policy_search.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv 
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc import HumanoidRandDirecEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_direc_2d import AntRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import  HumanoidRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from meta_policy_search.envs.mujoco_envs.swimmer_rand_vel import  SwimmerRandVelEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import  HalfCheetahRandDirecEnv
from meta_policy_search.envs.mujoco_envs.walker2d_rand_direc import Walker2DRandDirecEnv  
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_policy_search.envs.mujoco_envs.walker2d_rand_vel import Walker2DRandVelEnv

# from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline









modes = ["MAML on two envs", 
         "MultiMAML", 
         "KAML with no initialization", 
         "KAML with phi initialization", 
         "KAML with late theta initialization"]
    
mode = "KAML with late theta initialization"
print("using mode: ", mode)
hidden_sizes = (64,64)

meta_policy_search_path = '/'.join(os.path.realpath(
    os.path.dirname(__file__)).split('/')[:-1])


def main(config):
    assert len(config['probs']) == len(config['env'])
    set_seed(config['seed']) # TODO: what is this line doing?

    baseline = globals()[config['baseline']]()  # instantiate baseline
    env_names = "_".join(config['env'])
    
    envs = [globals()[env]() for env in config['env']]
    envs = [normalize(env) for env in envs]  # apply normalize wrapper to env
    print("\n\n\n\n\nenvs = {}\n\n\n\n\n".format(env_names))
    max_action_dim = np.max([env.action_space.shape[0] for env in envs])
    max_obs_dim = np.max([env.observation_space.shape[0] for env in envs])

    num_clusters_upper_lim = config['num_clusters_upper_lim'] #len(envs)

    # Create a policy for each theta.
    policies = [MetaGaussianMLPPolicy(
        name="meta-policy-{}".format(i),
        obs_dim=max_obs_dim,
        action_dim=max_action_dim,
        meta_batch_size=config['meta_batch_size'],
        hidden_sizes=config['hidden_sizes'],
    ) for i in range(num_clusters_upper_lim)]

    # Create a sampler for each environment. 
    print("creating a sampler for each env")
    samplers = [MetaSampler(
        env=env,
        # This batch_size is confusing
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        max_path_length=config['max_path_length'],
        meta_batch_size=int(config['meta_batch_size']),
        max_obs_dim=max_obs_dim,
        task_action_dim=env.action_space.shape[0],
        parallel=config['parallel'],
    ) for env in envs]

    print("create sample processor")
    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )
    
    # Create an algorithm for each theta. 
    print("create algorithms")
    algos = []
    for i in range(num_clusters_upper_lim):
        algos.append(TRPOMAML(
            policy=policies[i],
            step_size=config['step_size'],
            inner_type=config['inner_type'],
            inner_lr=config['inner_lr'],
            meta_batch_size=config['meta_batch_size'],
            num_inner_grad_steps=config['num_inner_grad_steps'],
            exploration=False,
        ))
        print("\n\nalgorithm {} created.\n\n".format(i))
        
    print("define trainer")
    trainer = KAML_Test_Trainer(
        algos=algos,
        policies=policies,
        envs=envs,
        env_names = env_names,
        samplers=samplers,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        probs=config['probs'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        theta_count=num_clusters_upper_lim,
        multi_maml = config['multi_maml'],
        phi_test = config['phi_test'],
        switch_thresh = config['switch_thresh'],
        mode_name = config['mode_name'],
    )
    print("start training")
    trainer.train()


if __name__ == "__main__":
    idx = int(time.time())
    assert mode in modes, "unknown mode"
    parser = argparse.ArgumentParser(
        description='ProMP: Proximal Meta-Policy Search')
    parser.add_argument('--config_file', type=str, default='',
                        help='json file with run specifications')
    parser.add_argument('--dump_path', type=str,
                        default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)
    
    parser.add_argument('--envs', type=str,
                        default="AntRandDirecEnv, HalfCheetahRandDirecEnv")


    args = parser.parse_args()
    envs = args.envs.replace(" ", "").split(",")

#     assert 1 == 2, "args.envs = {}".format(envs)
    logger.log("mode: {}".format(mode))
    
    if args.config_file:  # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)

    else:  # use default config
        if mode == "MAML on two envs":
            config = {
                'seed': 2,

                'baseline': 'LinearFeatureBaseline',

                'env': envs,# ['AntRandDirecEnv', 'HalfCheetahRandDirecEnv'],
                'probs': [0.5, 0.5],

                # sampler config
                'rollouts_per_meta_task': 20,
                'max_path_length': 100,
                'parallel': True,

                # sample processor config
                'discount': 0.99,
                'gae_lambda': 1,
                'normalize_adv': True,

                # policy config
                'hidden_sizes': hidden_sizes,
                'learn_std': True,  # whether to learn the standard deviation of the gaussian policy

                # E-MAML config
                'inner_lr': 0.1,  # adaptation step size
                'learning_rate': 1e-3,  # meta-policy gradient step size
                'step_size': 0.01,  # size of the TRPO trust-region
                'n_itr': 1001,  # number of overall training iterations
                'meta_batch_size': 40,  # number of sampled meta-tasks per iterations
                'num_inner_grad_steps': 1,  # number of inner / adaptation gradient steps
                'inner_type': 'log_likelihood',  # type of inner loss function used

                'multi_maml': False,
                'phi_test': False,
                'switch_thresh': 1000,
                'num_clusters_upper_lim': 1,
                'mode_name': str(mode),

            }
            assert len(config['probs']) == 2
            assert len(config['env']) == 2
            assert config['num_clusters_upper_lim'] == 1
            assert config['multi_maml'] == False
            assert config['phi_test'] == False
            
            
        elif mode == "KAML with no initialization":
            config = {
                'seed': 2,

                'baseline': 'LinearFeatureBaseline',

                'env': envs,
                'probs': [0.5, 0.5],

                # sampler config
                'rollouts_per_meta_task': 20,
                'max_path_length': 100,
                'parallel': True,

                # sample processor config
                'discount': 0.99,
                'gae_lambda': 1,
                'normalize_adv': True,

                # policy config
                'hidden_sizes': hidden_sizes,
                'learn_std': True,  # whether to learn the standard deviation of the gaussian policy

                # E-MAML config
                'inner_lr': 0.1,  # adaptation step size
                'learning_rate': 1e-3,  # meta-policy gradient step size
                'step_size': 0.01,  # size of the TRPO trust-region
                'n_itr': 1001,  # number of overall training iterations
                'meta_batch_size': 40,  # number of sampled meta-tasks per iterations
                'num_inner_grad_steps': 1,  # number of inner / adaptation gradient steps
                'inner_type': 'log_likelihood',  # type of inner loss function used

                'multi_maml': True,
                'phi_test': False,
                'switch_thresh': 0,
                'num_clusters_upper_lim': 2,
                'mode_name': str(mode),
            }
            
            assert len(config['probs']) == 2
            assert config['num_clusters_upper_lim'] == 2
            assert config['multi_maml'] == True
            assert config['phi_test'] == False
            assert config['switch_thresh'] < 5
            
        elif mode == "KAML with phi initialization":
            config = {
                'seed': 2,

                'baseline': 'LinearFeatureBaseline',

                'env': envs,
                'probs': [0.5, 0.5],

                # sampler config
                'rollouts_per_meta_task': 20,
                'max_path_length': 100,
                'parallel': True,

                # sample processor config
                'discount': 0.99,
                'gae_lambda': 1,
                'normalize_adv': True,

                # policy config
                'hidden_sizes': hidden_sizes,
                'learn_std': True,  # whether to learn the standard deviation of the gaussian policy

                # E-MAML config
                'inner_lr': 0.1,  # adaptation step size
                'learning_rate': 1e-3,  # meta-policy gradient step size
                'step_size': 0.01,  # size of the TRPO trust-region
                'n_itr': 1001,  # number of overall training iterations
                'meta_batch_size': 40,  # number of sampled meta-tasks per iterations
                'num_inner_grad_steps': 1,  # number of inner / adaptation gradient steps
                'inner_type': 'log_likelihood',  # type of inner loss function used

                'multi_maml': True,
                'phi_test': True,
                'switch_thresh': 300,
                'num_clusters_upper_lim': 2,
                'mode_name': str(mode),
            }
            assert len(config['probs']) == 2
            assert config['num_clusters_upper_lim'] == 2
            assert config['multi_maml'] == True
            assert config['phi_test'] == True
            assert config['switch_thresh'] > 200
            
        # Basically MultiMAML
        elif mode == "KAML with late theta initialization":
            config = {
                'seed': 2,

                'baseline': 'LinearFeatureBaseline',

                'env': envs, #['AntRandDirecEnv', 'SwimmerRandVelEnv'],
                'probs': [0.5, 0.5],

                # sampler config
                'rollouts_per_meta_task': 20,
                'max_path_length': 100,
                'parallel': True,

                # sample processor config
                'discount': 0.99,
                'gae_lambda': 1,
                'normalize_adv': True,

                # policy config
                'hidden_sizes': hidden_sizes,
                'learn_std': True,  # whether to learn the standard deviation of the gaussian policy

                # E-MAML config
                'inner_lr': 0.1,  # adaptation step size
                'learning_rate': 1e-3,  # meta-policy gradient step size
                'step_size': 0.01,  # size of the TRPO trust-region
                'n_itr': 31, #1001,  # number of overall training iterations
                'meta_batch_size': 40,  # number of sampled meta-tasks per iterations
                'num_inner_grad_steps': 1,  # number of inner / adaptation gradient steps
                'inner_type': 'log_likelihood',  # type of inner loss function used

                'multi_maml': True,
                'phi_test': False,
                'switch_thresh': 0,
                'num_clusters_upper_lim': 3,
                'mode_name': str(mode), 
            }
            assert len(config['probs']) == 2
#             assert config['num_clusters_upper_lim'] == 2
            assert config['multi_maml'] == True
            assert config['phi_test'] == False
            assert config['switch_thresh'] < 200
        else:
            assert 1 == 2, "Unknown mode {}".format(mode)

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv', 'tensorboard'],
                     snapshot_mode='last_gap')
    logger.log("Training for {}".format(mode))
    # dump run configuration before starting training
    json.dump(config, open(args.dump_path +
                           '/params.json', 'w'), cls=ClassEncoder)

    # start the actual algorithm
    main(config)
