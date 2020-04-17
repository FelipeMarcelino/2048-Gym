import numpy as np
import warnings
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear
from itertools import zip_longest


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].
    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if "pi" in layer:
                assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer["pi"]

            if "vf" in layer:
                assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer["vf"]
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value


def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    # layer 1
    conv1 = conv(scaled_images, "c1", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv2 = conv(scaled_images, "c2", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    relu1 = activ(conv1)
    relu2 = activ(conv2)
    # layer 2
    conv11 = conv(relu1, "c3", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv12 = conv(relu1, "c4", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv21 = conv(relu2, "c3", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv22 = conv(relu2, "c4", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    # layer2 relu activation
    relu11 = tf.nn.relu(conv11)
    relu12 = tf.nn.relu(conv12)
    relu21 = tf.nn.relu(conv21)
    relu22 = tf.nn.relu(conv22)

    # get shapes of all activations
    shape1 = relu1.get_shape().as_list()
    shape2 = relu2.get_shape().as_list()

    shape11 = relu11.get_shape().as_list()
    shape12 = relu12.get_shape().as_list()
    shape21 = relu21.get_shape().as_list()
    shape22 = relu22.get_shape().as_list()

    # expansion
    hidden1 = tf.reshape(relu1, [-1, shape1[1] * shape1[2] * shape1[3]])
    hidden2 = tf.reshape(relu2, [-1, shape2[1] * shape2[2] * shape2[3]])

    hidden11 = tf.reshape(relu11, [-1, shape11[1] * shape11[2] * shape11[3]])
    hidden12 = tf.reshape(relu12, [-1, shape12[1] * shape12[2] * shape12[3]])
    hidden21 = tf.reshape(relu21, [-1, shape21[1] * shape21[2] * shape21[3]])
    hidden22 = tf.reshape(relu22, [-1, shape22[1] * shape22[2] * shape22[3]])

    # concatenation
    hidden = tf.concat([hidden1, hidden2, hidden11, hidden12, hidden21, hidden22], axis=1)

    # linear layer 1
    linear_1 = activ(linear(hidden, scope="fc1", n_hidden=512, init_scale=np.sqrt(2)))

    # linear layer 2
    linear_2 = activ(linear(linear_1, scope="fc2", n_hidden=128, init_scale=np.sqrt(2)))

    return linear_2


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        reuse=False,
        layers=None,
        net_arch=None,
        act_fun=tf.tanh,
        cnn_extractor=modified_cnn,
        feature_extraction="cnn",
        **kwargs
    ):
        super(FeedForwardPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=(feature_extraction == "cnn")
        )

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn(
                "Usage of the `layers` parameter is deprecated! Use net_arch instead "
                "(it has a different semantics though).",
                DeprecationWarning,
            )
            if net_arch is not None:
                warnings.warn(
                    "The new `net_arch` parameter overrides the deprecated `layers` parameter!", DeprecationWarning
                )

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, "vf", 1)

            self._proba_distribution, self._policy, self.q_value = self.pdtype.proba_distribution_from_latent(
                pi_latent, vf_latent, init_scale=0.01
            )

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp], {self.obs_ph: obs}
            )
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class CustomCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomCnnPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, feature_extraction="cnn", **_kwargs
        )


class CustomMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    # Trying a shared layer here
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomMlpPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse,
            feature_extraction="mlp",
            net_arch=[512, dict(pi=[256, 128, 64], vf=[256, 128, 64])],
            **_kwargs,
        )
