"""RSL-RL PPO configuration for MaxArm-Charger-Align-v0.

This follows Isaac Lab's standard RSL-RL config style. If your Isaac Lab version
has slightly different config field names, compare this file with one of the
built-in direct tasks in:
    source/isaaclab_tasks/isaaclab_tasks/direct/*/agents/
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MaxArmChargerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 100
    experiment_name = "maxarm_charger_align"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
