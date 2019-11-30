# Cheetah
import numpy as np
from meta_policy_search.envs.base import MetaEnv
from meta_policy_search.utils import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

# Sawyer
from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv as SawyerEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
import numpy as np



class HalfCheetahRandDirecEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
class SawyerPickAndPlaceEnv(FlatGoalEnv, MetaEnv)
    def __init__(self, goal_direction=None):
        self.mujoco_weight = 0.5
        # Cheetah
        self.cheetah_goal_direction = goal_direction if goal_direction else 1.0
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        gym.utils.EzPickle.__init__(self, goal_direction)
        # Sawyer
        self.sawyer_quick_init(locals())
        sawyer_env = SawyerEnv(*args, **kwargs)
        FlatGoalEnv.__init__(self, sawyer_env, obs_keys=['state_observation'], goal_keys=['state_desired_goal'])

    
        
    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        
        return np.random.choice((-1.0, 1.0), (n_tasks, )) # Cheetah
        return self.sawyer_sample_goals(n_tasks) # Sawyer
    
    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        
        self.cheetah_goal_direction = task # Cheetah
        return self.sawyer_set_goal(task) # Sawyer
        
    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.cheetah_goal_direction # Cheetah
        return self.sawyer_get_goal() # Sawyer

    def step(self, action):
        xposbefore = self.cheetah_sim.data.qpos[0]
        self.cheetah_do_simulation(action, self.cheetah_frame_skip)
        xposafter = self.cheetah_sim.data.qpos[0]
        ob = self.cheetah__get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        reward_run = self.cheetah_goal_direction * (xposafter - xposbefore) / self.cheetah_dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.cheetah_sim.data.qpos.flat[1:],
            self.cheetah_sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.cheetah_init_qpos + self.cheetah_np_random.uniform(low=-.1, high=.1, size=self.cheetah_model.nq)
        qvel = self.cheetah_init_qvel + self.cheetah_np_random.randn(self.cheetah_model.nv) * .1
        self.cheetah_set_state(qpos, qvel)
        return self.cheetah__get_obs()

    def viewer_setup(self):
        self.cheetah_viewer.cam.distance = self.cheetah_model.stat.extent * 0.5

    
    
    def log_diagnostics(self, paths, prefix=''):
        try:
            # Cheetah Diag
            fwrd_vel = [path["env_infos"]['reward_run'] for path in paths]
            final_fwrd_vel = [path["env_infos"]['reward_run'][-1] for path in paths]
            ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

            logger.logkv(prefix + 'AvgForwardVel', np.mean(fwrd_vel))
            logger.logkv(prefix + 'AvgFinalForwardVel', np.mean(final_fwrd_vel))
            logger.logkv(prefix + 'AvgCtrlCost', np.std(ctrl_cost))
        except:
            # Sawyer Diag
            reach_dist = [path["env_infos"]['reachDist'] for path in paths]
            placing_dist = [path["env_infos"]['placeDist'] for path in paths]

            logger.logkv(prefix + 'AverageReachDistance', np.mean(reach_dist))
            logger.logkv(prefix + 'AveragePlaceDistance', np.mean(placing_dist))

    def __str__(self):
        return 'HalfCheetahRandDirecEnv'
    
# Sawyer Extra Functions

#     def render(self):
#         SawyerEnv.render(self)
#     @property
#     def action_space(self):
#         return FlatGoalEnv.action_space(self)

        
    