import collections

from maml_zoo.logger import logger
from maml_zoo.samplers.base import SampleProcessor
import numpy as np

class RL2SampleProcessor(SampleProcessor):

    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []


        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, compute advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)
            # samples_data['extended_obs'] = np.concatenate([samples_data['observations'], samples_data['actions'],
            #                                                samples_data['rewards'], samples_data['dones']], axis=-1)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

            

        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos = \
            self._stack_path_data(samples_data_meta_batch)

        overall_avg_reward = np.mean(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std = np.std(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))

        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        average_success_rate = np.mean(np.stack([np.any(p['env_infos']['success'] == 1) for p in all_paths]))
        logger.logkv(log_prefix + 'AverageSuccessRate', average_success_rate)

        if 'task' in all_paths[0]['env_infos']:
            tasks = [int(np.nonzero(p['env_infos']['task'][0,:])[0]) for p in all_paths]
            success_rates = collections.defaultdict(lambda: [])
            avg_disc_returns = collections.defaultdict(lambda: [])
            import ipdb
            ipdb.set_trace()
            for t, p in zip(tasks, all_paths):
                success_rates[t].append(np.any(p['env_infos']['success'] == 1))
                avg_disc_returns[t].append(p['returns'][0])
                
            for t in success_rates.keys():
                logger.logkv('{}TaskAverageSuccessRate-{}'.format(log_prefix, t), np.mean(np.array(success_rates[t])))
                logger.logkv('{}TaskAverageDiscountedReturn-{}'.format(log_prefix, t), np.mean(np.array(avg_disc_returns[t])))

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        return samples_data
