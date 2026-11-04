"""MaxArm charger task package.

The Gymnasium registration block is intentionally guarded so offline utilities
(such as FK checks) can run outside the Isaac Lab environment.
"""

try:
    import gymnasium as gym

    gym.register(
        id="MaxArm-Charger-Align-v0",
        entry_point="maxarm_charger.tasks.direct.maxarm_charger_align_env:MaxArmChargerAlignEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "maxarm_charger.tasks.direct.maxarm_charger_align_env:MaxArmChargerAlignEnvCfg",
            "rsl_rl_cfg_entry_point": "maxarm_charger.agents.rsl_rl_ppo_cfg:MaxArmChargerPPORunnerCfg",
        },
    )
except Exception:
    # Outside Isaac/Gymnasium, registration is not needed.
    pass
