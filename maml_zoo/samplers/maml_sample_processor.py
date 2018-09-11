from maml_zoo.samplers.base import SampleProcessor
from maml_zoo.samplers.dice_sample_processor import DiceSampleProcessor
from maml_zoo.utils import utils
import numpy as np

class MAMLSampleProcessor(SampleProcessor):
    def __init__(self, normalize_adv_per_task=True, *args, **kwargs):
        self.normalize_adv_per_task = normalize_adv_per_task
        super(MAMLSampleProcessor, self).__init__(*args, **kwargs)

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

        if not self.normalize_adv_per_task:
            _normalize_adv = self.normalize_adv
            self.normalize_adv = False

        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, compute advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        if not self.normalize_adv_per_task:
            advantages = np.stack([samples_data['advantages'] for samples_data in samples_data_meta_batch])
            if _normalize_adv:
                advantages = utils.normalize_advantages(advantages)
                for i in range(len(samples_data_meta_batch)):
                    samples_data_meta_batch[i]['advantages'] = advantages[i]
            self.normalize_adv = _normalize_adv

        # 7) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        return samples_data_meta_batch

class DiceMAMLSampleProcessor(DiceSampleProcessor):
    process_samples = MAMLSampleProcessor.process_samples