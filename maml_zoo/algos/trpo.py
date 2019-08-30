from maml_zoo.logger import logger
from maml_zoo.algos.base import Algo
from maml_zoo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


import tensorflow as tf
from collections import OrderedDict


class TRPO(Algo):
    """
    Algorithm for TRPO

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        exploration (bool): use exploration / pre-update sampling term / E-MAML term
        inner_type (str): inner optimization objective - either log_likelihood or likelihood_ratio
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            *args,
            name="trpo",
            step_size=0.01,
            max_epochs=5,
            **kwargs
            ):
        super(TRPO, self).__init__(*args, **kwargs)

        self.recurrent = getattr(self.policy, 'recurrent', False)
        self.optimizer = ConjugateGradientOptimizer()
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self._step_size = step_size

        self.build_graph()

    def build_graph(self):
        """
        Creates the computation graph

        Notes:
            Pseudocode:
            for task in meta_batch_size:
                make_vars
                init_init_dist_sym
            for step in num_inner_grad_steps:
                for task in meta_batch_size:
                    make_vars
                    update_init_dist_sym
            set objectives for optimizer
        """

        """ Create Variables """

        """ ----- Build graph for the meta-update ----- """
        self.meta_op_phs_dict = OrderedDict()
        obs_ph, action_ph, adv_ph, dist_info_old_ph, all_phs_dict = self._make_input_placeholders('train',
                                                                                                  recurrent=self.recurrent)
        self.meta_op_phs_dict.update(all_phs_dict)

        # dist_info_vars_for_next_step
        if self.recurrent:
            distribution_info_vars, hidden_ph, next_hidden_var = self.policy.distribution_info_sym(obs_ph)
        else:
            distribution_info_vars = self.policy.distribution_info_sym(obs_ph)
            hidden_ph, next_hidden_var = None, None

        """kl divergence"""
        kl = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_ph, distribution_info_vars))

        """ Outer objective """
        # meta-objective
        likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_ph, dist_info_old_ph,
                                                                         distribution_info_vars)

        objective = likelihood_ratio * adv_ph
        surr_obj = - tf.reduce_mean(objective)

        self.optimizer.build_graph(
            loss=surr_obj,
            target=self.policy,
            input_ph_dict=self.meta_op_phs_dict,
            leq_constraint=(kl, self._step_size),
        )

    def optimize_policy(self, samples_data, log=True):
        """
        Performs MAML outer step

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix='train')

        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(input_dict)

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_dict)
        logger.log("Optimizing")
        self.optimizer.optimize(input_dict)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_dict)

        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(input_dict)
        if log:
            logger.logkv('MeanKLBefore', mean_kl_before)
            logger.logkv('MeanKL', mean_kl)

            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('dLoss', loss_before - loss_after)
