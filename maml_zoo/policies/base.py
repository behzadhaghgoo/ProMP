from maml_zoo.utils.utils import remove_scope_from_name
from maml_zoo.utils import Serializable
import tensorflow as tf
from collections import OrderedDict


class Policy(Serializable):
    """
    A container for storing the current pre and post update policies
    Also provides functions for executing and updating policy parameters

    Note:
        the preupdate policy is stored as tf.Variables, while the postupdate
        policy is stored in numpy arrays and executed through tf.placeholders

    Args:
        obs_dim (int): dimensionality of the observation space -> specifies the input size of the policy
        action_dim (int): dimensionality of the action space -> specifies the output size of the policy
        name (str) : Name used for scoping variables in policy
        hidden_sizes (tuple) : size of hidden layers of network
        learn_std (bool) : whether to learn variance of network output
        hidden_nonlinearity (Operation) : nonlinearity used between hidden layers of network
        output_nonlinearity (Operation) : nonlinearity used after the final layer of network
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 name='policy',
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 hidden_nonlinearity=tf.tanh,
                 output_nonlinearity=None,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name

        self.hidden_sizes = hidden_sizes
        self.learn_std = learn_std
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self._dist = None
        self.policy_params = None
        self._assign_ops = None
        self._assign_phs = None

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        raise NotImplementedError

    def get_action(self, observation):
        """
        Runs a single observation through the specified policy

        Args:
            observation (array) : single observation

        Returns:
            (array) : array of arrays of actions for each env
        """
        raise NotImplementedError

    def get_actions(self, observations):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (array) : array of arrays of observations generated by each task and env

        Returns:
            (tuple) : array of arrays of actions for each env (meta_batch_size) x (batch_size) x (action_dim)
                      and array of arrays of agent_info dicts 
        """
        raise NotImplementedError

    def reset(self, dones=None):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def distribution(self):
        """
        Returns this policy's distribution

        Returns:
            (Distribution) : this policy's distribution
        """
        raise NotImplementedError

    def distribution_info_sym(self, obs_var, params=None):
        """
        Return the symbolic distribution information about the actions.

        Args:
            obs_var (placeholder) : symbolic variable for observations
            params (None or dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise NotImplementedError

    def distribution_info_keys(self, obs, state_infos):
        """
        Args:
            obs (placeholder) : symbolic variable for observations
            state_infos (dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise NotImplementedError

    def likelihood_ratio_sym(self, obs, action, dist_info_old, policy_params):
        """
        Computes the likelihood p_new(obs|act)/p_old ratio between

        Args:
            obs (tf.Tensor): symbolic variable for observations
            action (tf.Tensor): symbolic variable for actions
            dist_info_old (dict): dictionary of tf.placeholders with old policy information
            policy_params (dict): dictionary of the policy parameters (each value is a tf.Tensor)

        Returns:
            (tf.Tensor) : likelihood ratio
        """

        distribution_info_new = self.distribution_info_sym(obs, params=policy_params)
        likelihood_ratio = self._dist.likelihood_ratio_sym(action, dist_info_old, distribution_info_new)
        return likelihood_ratio

    def log_likelihood_sym(self, obs, action, policy_params):
        """
        Computes the log likelihood p(obs|act)

        Args:
            obs (tf.Tensor): symbolic variable for observations
            action (tf.Tensor): symbolic variable for actions
            policy_params (dict): dictionary of the policy parameters (each value is a tf.Tensor)

        Returns:
            (tf.Tensor) : log likelihood
        """

        distribution_info_var = self._dist.distribution_info_sym(obs, params=policy_params)
        log_likelihood = self._dist.log_likelihood_sym(action, distribution_info_var)
        return log_likelihood

    """ --- methods for serialization --- """

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self.policy_params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        param_values = tf.get_default_session().run(self.policy_params)
        return param_values

    def set_params(self, policy_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), policy_params.keys())]), \
            "parameter keys must match with variable"

        if self._assign_ops is None:
            assign_ops, assign_phs = [], []
            for var in self.get_params().values():
                assign_placeholder = tf.placeholder(dtype=var.dtype)
                assign_op = tf.assign(var, assign_placeholder)
                assign_ops.append(assign_op)
                assign_phs.append(assign_placeholder)
            self._assign_ops = assign_ops
            self._assign_phs = assign_phs
        feed_dict = dict(zip(self._assign_phs, policy_params.values()))
        tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

    def __getstate__(self):
        state = {
            'init_args': Serializable.__getstate__(self),
            'network_params': self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        state['init_args']['__kwargs']['cell_type'] = 'lstm'
        Serializable.__setstate__(self, state['init_args'])
        tf.get_default_session().run(tf.global_variables_initializer())
        self.set_params(state['network_params'])
        self.state = state


class MetaPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super(MetaPolicy, self).__init__(*args, **kwargs)
        self._pre_update_mode = True
        self.policies_params_vals = None
        self.policy_params_keys = None
        self.policies_params_phs = None
        self.meta_batch_size = None

    def build_graph(self):
        """
        Also should create lists of variables and corresponding assign ops
        """
        raise NotImplementedError

    def switch_to_pre_update(self):
        """
        Switches get_action to pre-update policy
        """
        self._pre_update_mode = True
        # replicate pre-update policy params meta_batch_size times
        self.policies_params_vals = [self.get_param_values() for _ in range(self.meta_batch_size)]

    def get_actions(self, observations):
        if self._pre_update_mode:
            return self._get_pre_update_actions(observations)
        else:
            return self._get_post_update_actions(observations)

    def _get_pre_update_actions(self, observations):
        """
        Args:
            observations (list): List of size meta-batch size with numpy arrays of shape batch_size x obs_dim
        """
        raise NotImplementedError

    def _get_post_update_actions(self, observations):
        """
        Args:
            observations (list): List of size meta-batch size with numpy arrays of shape batch_size x obs_dim
        """
        raise NotImplementedError

    def update_task_parameters(self, updated_policies_parameters):
        """
        Args:
            updated_policies_parameters (list): List of size meta-batch size. Each contains a dict with the policies
            parameters as numpy arrays
        """
        self.policies_params_vals = updated_policies_parameters
        self._pre_update_mode = False

    def _create_placeholders_for_vars(self, scope, graph_keys=tf.GraphKeys.TRAINABLE_VARIABLES):
        var_list = tf.get_collection(graph_keys, scope=scope)
        placeholders = []
        for var in var_list:
            var_name = remove_scope_from_name(var.name, scope.split('/')[0])
            placeholders.append((var_name, tf.placeholder(tf.float32, shape=var.shape, name="%s_ph" % var_name)))
        return OrderedDict(placeholders)

    @property
    def policies_params_feed_dict(self):
        """
            returns fully prepared feed dict for feeding the currently saved policy parameter values
            into the lightweight policy graph
        """
        return dict(list((self.policies_params_phs[i][key], self.policies_params_vals[i][key])
                         for key in self.policy_params_keys for i in range(self.meta_batch_size)))


