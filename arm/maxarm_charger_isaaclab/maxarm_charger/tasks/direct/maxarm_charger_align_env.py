"""State-based alignment environment for the MaxArm charger task.

This version is intentionally conservative. It fixes the robot root, avoids
absolute IK branch jumps, and provides explicit debug modes so the physical
articulation can be validated before training.

Recommended validation order:
    MAXARM_DEBUG_MODE=hold          -> force-freeze joints every step; robot must stay still
    MAXARM_DEBUG_MODE=drive_hold    -> hold the fixed reset pose using position drives only
    MAXARM_DEBUG_MODE=fixed_neg_y   -> robot should move smoothly in one direction
    unset MAXARM_DEBUG_MODE         -> PPO/policy controls the small Cartesian action
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence

import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This file must be imported from an Isaac Lab Python environment. "
        "Run it with ./isaaclab.sh -p or inside the Isaac Lab environment."
    ) from exc

from maxarm_charger.utils.maxarm_math import (
    JOINT_LOWER,
    JOINT_UPPER,
    analytic_ik_position,
    differential_ik_step,
)
from maxarm_charger.utils.debug_helpers import DebugDrawHelper, TrajectoryCSVLogger


@configclass
class MaxArmChargerAlignEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # RL timing
    # ------------------------------------------------------------------
    episode_length_s = 6.0
    decimation = 4
    action_space = 3
    observation_space = 12
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    # Parallel rollouts. The USD was already validated single-env; PPO needs many
    # parallel envs (~thousands of samples per iter) for the policy gradient to
    # produce useful signal in 3D continuous control.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=1.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # ------------------------------------------------------------------
    # Robot asset
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/MaxArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/salman/Documents/simsim/newest_work_17_5.usd",
            # The charging stage assumes the mobile base is stopped. The arm root
            # must not be free in physics, otherwise it can rotate/fall as a floating body.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "after_base_full_link_joint": 0.0,
                "link1_link_joint": 0.0,
                "link2_link_joint": 0.0,
                "end_effector_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        # Configure every joint so the fourth USD joint is not left without a drive.
        # The RL policy still commands only the three active joints below.
        actuators={
            "all_position_drives": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # Idealized high-gain servo drive for debugging the imported USD.
                # Your real Hiwonder servos are internally position-controlled; during
                # Phase 1 we want the articulation to behave like a commanded servo arm,
                # not like a passive chain falling under gravity.
                effort_limit_sim=100.0,
                velocity_limit_sim=20.0,
                stiffness=5000.0,
                damping=150.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )

    arm_joint_names = [
        "after_base_full_link_joint",
        "link1_link_joint",
        "link2_link_joint",
    ]
    passive_joint_names = ["end_effector_joint"]

    # Your USD exposed bodies: ['arm_base_link', 'after_base_full_link', 'link1_link', 'link2_link', 'visuals'].
    # 'visuals' is closer to the drawn end effector in your current file. If it fails,
    # the environment automatically falls back to link2_link.
    tip_body_name = "visuals"
    tip_body_fallback_names = ["visuals", "link2_link"]

    # ------------------------------------------------------------------
    # Task geometry, in arm-base/world frame for env_0.
    # ------------------------------------------------------------------
    socket_nominal_pos = (0.0, -0.10, 0.10)
    preinsert_distance = 0.03

    # Start with a fixed target. Randomization comes after the robot is stable.
    socket_x_range = (0.0, 0.0)
    socket_y_range = (-0.10, -0.10)
    socket_z_range = (0.10, 0.10)

    # Cartesian action authority per policy step. With episode_length_s=6.0 and
    # env step = 1/30s, an episode is ~180 steps. The arm must cover the ~16cm
    # distance from the zero-joint start pose to the pre-insert target within
    # that horizon, so the per-step delta needs to be on the order of millimeters,
    # not sub-millimeter.
    max_delta_x = 0.005
    max_delta_y = 0.005
    max_delta_z = 0.005
    max_joint_step = math.radians(3.0)
    ik_damping = 5.0e-3

    # Reset behavior. Use 'zero' while the USD/DH convention is still being validated.
    # Set MAXARM_RESET_MODE=analytic only after FK/IK is confirmed against Isaac.
    reset_mode = "zero"  # 'zero' or 'analytic'
    start_offset_y = 0.020

    success_pos_error = 0.003
    success_hold_steps = 8
    wall_margin = 0.004

    w_pos = 8.0
    w_action = 0.005
    w_smooth = 0.10
    w_joint_speed = 0.02
    w_success = 20.0
    w_joint_limit = 2.0
    w_wall_cross = 15.0
    w_ik_invalid = 5.0
    # Dense close-up bonus: amplifies the reward gradient as the tip approaches
    # the target so PPO has a strong "keep going" signal in the last few cm,
    # not just the linear -w_pos*d slope that flattens near zero.
    w_close = 5.0
    close_scale = 0.02

    # Debug modes:
    #   policy       : normal PPO action
    #   hold         : bypass PPO and IK; forcibly write frozen joint state every step
    #   drive_hold   : bypass PPO and IK; hold using position-drive targets only
    #   zero_delta   : zero Cartesian action but still passes through differential IK
    #   fixed_neg_y  : constant negative-y Cartesian action
    #   fixed_pos_y  : constant positive-y Cartesian action
    debug_control_mode = "policy"

    debug_draw = True
    debug_print = True
    debug_csv = True
    debug_print_action_every = 20
    debug_print_reward_every = 50
    debug_csv_dir = "logs/debug_trajectories"


class MaxArmChargerAlignEnv(DirectRLEnv):
    cfg: MaxArmChargerAlignEnvCfg

    def __init__(self, cfg: MaxArmChargerAlignEnvCfg, render_mode: str | None = None, **kwargs):
        self.actions = None
        self.prev_actions = None
        self.q_target = None
        self.q_target_all = None
        self.target_pos_w = None
        self.socket_pos_w = None
        self.success_counter = None
        self.ik_valid = None
        self.arm_joint_ids = None
        self.passive_joint_ids = None
        self.tip_body_id = None
        self._hold_joint_pos_all = None
        self._hold_root_state = None

        self._global_step = 0
        self._episode_id = 0
        self._episode_step = 0
        self._episode_rewards: list[float] = []
        self._episode_errors: list[float] = []
        self._last_reward_breakdown: dict = {}
        self._last_desired_delta_p = None

        self._debug_draw_helper = DebugDrawHelper(enabled=cfg.debug_draw)
        self._csv_logger = TrajectoryCSVLogger(enabled=cfg.debug_csv, log_dir=cfg.debug_csv_dir)

        super().__init__(cfg, render_mode, **kwargs)

        self._initialize_robot_handles_and_buffers()
        self._announce_play_mode()
        self._print_model_summary()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_robot_handles_and_buffers(self):
        if self.arm_joint_ids is not None:
            return

        self.arm_joint_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        try:
            self.passive_joint_ids, _ = self.robot.find_joints(self.cfg.passive_joint_names)
        except Exception:
            self.passive_joint_ids = []

        body_error = None
        for name in self.cfg.tip_body_fallback_names:
            try:
                self.tip_body_id, _ = self.robot.find_bodies(name)
                self.cfg.tip_body_name = name
                break
            except Exception as exc:
                body_error = exc
                self.tip_body_id = None
        if self.tip_body_id is None:
            raise RuntimeError(
                f"Could not find any tip body from {self.cfg.tip_body_fallback_names}. "
                f"Available bodies: {self.robot.body_names}. Last error: {body_error}"
            )

        if len(self.arm_joint_ids) != 3:
            raise RuntimeError(
                f"Expected 3 active joints, got {len(self.arm_joint_ids)}. "
                f"Available joints: {self.robot.joint_names}"
            )

        self.actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.q_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.q_target_all = torch.zeros_like(self.robot.data.joint_pos)
        self.target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.socket_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.success_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ik_valid = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _announce_play_mode(self) -> None:
        script_name = os.path.basename(sys.argv[0]).lower() if sys.argv else ""
        if "play" not in script_name:
            return
        checkpoint = None
        for i, a in enumerate(sys.argv):
            if a == "--checkpoint" and i + 1 < len(sys.argv):
                checkpoint = sys.argv[i + 1]
            elif a.startswith("--checkpoint="):
                checkpoint = a.split("=", 1)[1]
        print("\n[MaxArm Debug] " + "=" * 60)
        print(f"[MaxArm Debug] PLAY MODE DETECTED (script: {script_name})")
        print(f"[MaxArm Debug] Checkpoint: {checkpoint if checkpoint else '(none detected)'}")
        if checkpoint:
            print(f"[MaxArm Debug] Checkpoint exists: {os.path.exists(checkpoint)}")
        else:
            print("[MaxArm Debug] WARNING: no explicit checkpoint detected; policy may be random.")
        print("[MaxArm Debug] " + "=" * 60 + "\n")

    def _print_model_summary(self) -> None:
        print("\n[MaxArm Debug] Robot summary")
        print(f"  joints        : {self.robot.joint_names}")
        print(f"  bodies        : {self.robot.body_names}")
        print(f"  active ids    : {self.arm_joint_ids}")
        print(f"  passive ids   : {self.passive_joint_ids}")
        print(f"  tip body      : {self.cfg.tip_body_name} -> {self.tip_body_id}")
        print(f"  reset mode    : {os.getenv('MAXARM_RESET_MODE', self.cfg.reset_mode)}")
        print(f"  control mode  : {os.getenv('MAXARM_DEBUG_MODE', self.cfg.debug_control_mode)}\n")

    def _control_mode(self) -> str:
        return os.getenv("MAXARM_DEBUG_MODE", self.cfg.debug_control_mode).strip().lower()

    def _reset_mode(self) -> str:
        return os.getenv("MAXARM_RESET_MODE", self.cfg.reset_mode).strip().lower()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._initialize_robot_handles_and_buffers()

        self.prev_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)

        q_current_all = self.robot.data.joint_pos.clone()
        q_current = q_current_all[:, self.arm_joint_ids]

        mode = self._control_mode()
        if mode == "hold":
            # Strongest sanity test: bypass PPO, IK, and physics drift.
            # We write the same joint state back into the simulator every step. If links
            # still fall in this mode, the falling object is not actually part of the
            # articulation or the USD hierarchy is broken.
            self.actions[:] = 0.0
            if self._hold_joint_pos_all is None:
                self._hold_joint_pos_all = q_current_all.clone()
            hold_q = self._hold_joint_pos_all.clone()
            hold_dq = torch.zeros_like(hold_q)
            self.robot.write_joint_state_to_sim(hold_q, hold_dq)
            self.robot.set_joint_position_target(hold_q)
            self.q_target_all = hold_q
            self.q_target = hold_q[:, self.arm_joint_ids]
            self.ik_valid[:] = True
            
            return
        if mode == "drive_hold":
            # Drive-only hold. This tests whether the configured position drives are
            # strong enough to hold the links under gravity.
            # IMPORTANT: the target must be the frozen reset pose, not q_current.
            # If we use q_current every step, the controller follows the falling arm
            # and generates almost no restoring torque.
            self.actions[:] = 0.0
            if self._hold_joint_pos_all is None:
                self._hold_joint_pos_all = q_current_all.clone()
            hold_q = self._hold_joint_pos_all.clone()
            self.q_target_all = hold_q
            self.q_target = hold_q[:, self.arm_joint_ids]
            self.ik_valid[:] = True
            return
        if mode == "zero_delta":
            self.actions[:] = 0.0
        elif mode == "fixed_neg_y":
            self.actions[:] = 0.0
            self.actions[:, 1] = -1.0
        elif mode == "fixed_pos_y":
            self.actions[:] = 0.0
            self.actions[:, 1] = 1.0
        
        delta_p = torch.zeros((self.num_envs, 3), device=self.device)
        delta_p[:, 0] = self.actions[:, 0] * self.cfg.max_delta_x
        delta_p[:, 1] = self.actions[:, 1] * self.cfg.max_delta_y
        delta_p[:, 2] = self.actions[:, 2] * self.cfg.max_delta_z
        self._last_desired_delta_p = delta_p

        # Use differential IK. It produces a small local joint update and avoids
        # the 180-degree branch jump that analytic IK can produce for the same point.
        self.q_target = differential_ik_step(
            q_current=q_current,
            delta_p=delta_p,
            damping=self.cfg.ik_damping,
            max_joint_step=self.cfg.max_joint_step,
        )
        self.ik_valid[:] = torch.isfinite(self.q_target).all(dim=-1)
        self.q_target = torch.where(self.ik_valid.unsqueeze(-1), self.q_target, q_current)

        self.q_target_all = q_current_all.clone()
        self.q_target_all[:, self.arm_joint_ids] = self.q_target
        if self.passive_joint_ids:
            # Keep the fourth/end-effector joint fixed at zero unless you later add the
            # real mimic relation for the parallel mechanism.
            for jid in self.passive_joint_ids:
                self.q_target_all[:, jid] = 0.0

        if self.cfg.debug_draw and self._debug_draw_helper.enabled:
            tip_pos_w = self.robot.data.body_pos_w[:, self.tip_body_id[0], :]
            self._debug_draw_helper.draw_tip_target(
                tip_pos_w[0].detach().cpu().tolist(),
                self.target_pos_w[0].detach().cpu().tolist(),
            )

        if (
            self.cfg.debug_print
            and self.cfg.debug_print_action_every > 0
            and self._global_step % self.cfg.debug_print_action_every == 0
        ):
            raw = actions[0].detach().cpu().tolist()
            act = self.actions[0].detach().cpu().tolist()
            dp = delta_p[0].detach().cpu().tolist()
            qc = q_current[0].detach().cpu().tolist()
            qt = self.q_target[0].detach().cpu().tolist()
            print(
                f"\n--- ACTION DEBUG (step {self._global_step}, mode={mode}) ---\n"
                f"  raw action     : [{raw[0]:+.3f}, {raw[1]:+.3f}, {raw[2]:+.3f}]\n"
                f"  used action    : [{act[0]:+.3f}, {act[1]:+.3f}, {act[2]:+.3f}]\n"
                f"  delta_p (mm)   : [{1000*dp[0]:+.3f}, {1000*dp[1]:+.3f}, {1000*dp[2]:+.3f}]\n"
                f"  q_current(deg) : [{math.degrees(qc[0]):+.2f}, {math.degrees(qc[1]):+.2f}, {math.degrees(qc[2]):+.2f}]\n"
                f"  q_target (deg) : [{math.degrees(qt[0]):+.2f}, {math.degrees(qt[1]):+.2f}, {math.degrees(qt[2]):+.2f}]\n"
                f"  IK valid env0  : {bool(self.ik_valid[0].item())}"
            )

    def _apply_action(self):
        self._initialize_robot_handles_and_buffers()
        # Command all joints so the passive/end-effector joint does not drift.
        self.robot.set_joint_position_target(self.q_target_all)

    # ------------------------------------------------------------------
    # Observations / rewards / dones
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._initialize_robot_handles_and_buffers()
        tip_pos_w = self.robot.data.body_pos_w[:, self.tip_body_id[0], :]
        pos_error = self.target_pos_w - tip_pos_w
        q = self.robot.data.joint_pos[:, self.arm_joint_ids]
        dq = self.robot.data.joint_vel[:, self.arm_joint_ids]
        obs = torch.cat([pos_error, q, dq, self.prev_actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._initialize_robot_handles_and_buffers()
        tip_pos_w = self.robot.data.body_pos_w[:, self.tip_body_id[0], :]
        pos_error = self.target_pos_w - tip_pos_w
        pos_norm = torch.linalg.norm(pos_error, dim=-1)

        q = self.robot.data.joint_pos[:, self.arm_joint_ids]
        dq = self.robot.data.joint_vel[:, self.arm_joint_ids]
        action_norm = torch.linalg.norm(self.actions, dim=-1)
        smooth_norm = torch.linalg.norm(self.actions - self.prev_actions, dim=-1)
        dq_norm = torch.linalg.norm(dq, dim=-1)

        success_now = pos_norm < self.cfg.success_pos_error
        self.success_counter = torch.where(success_now, self.success_counter + 1, torch.zeros_like(self.success_counter))
        success_stable = self.success_counter >= self.cfg.success_hold_steps

        joint_lower = JOINT_LOWER.to(device=self.device, dtype=q.dtype)
        joint_upper = JOINT_UPPER.to(device=self.device, dtype=q.dtype)
        margin = math.radians(4.0)
        lower_violation = torch.relu((joint_lower + margin) - q)
        upper_violation = torch.relu(q - (joint_upper - margin))
        joint_limit_penalty = torch.sum(lower_violation + upper_violation, dim=-1)

        wall_cross = tip_pos_w[:, 1] < (self.socket_pos_w[:, 1] - self.cfg.wall_margin)
        ik_invalid = ~self.ik_valid

        r_pos = -self.cfg.w_pos * pos_norm
        r_close = self.cfg.w_close * torch.exp(-pos_norm / self.cfg.close_scale)
        r_action = -self.cfg.w_action * action_norm
        r_smooth = -self.cfg.w_smooth * smooth_norm
        r_joint_speed = -self.cfg.w_joint_speed * dq_norm
        r_joint_limit = -self.cfg.w_joint_limit * joint_limit_penalty
        r_wall = -self.cfg.w_wall_cross * wall_cross.float()
        r_ik = -self.cfg.w_ik_invalid * ik_invalid.float()
        r_success = self.cfg.w_success * success_stable.float()
        reward = r_pos + r_close + r_action + r_smooth + r_joint_speed + r_joint_limit + r_wall + r_ik + r_success

        breakdown = {
            "pos": float(r_pos.mean().item()),
            "close": float(r_close.mean().item()),
            "action": float(r_action.mean().item()),
            "smooth": float(r_smooth.mean().item()),
            "joint_speed": float(r_joint_speed.mean().item()),
            "joint_limit": float(r_joint_limit.mean().item()),
            "wall": float(r_wall.mean().item()),
            "ik": float(r_ik.mean().item()),
            "success": float(r_success.mean().item()),
            "total": float(reward.mean().item()),
            "error_norm_m": float(pos_norm.mean().item()),
        }
        self._last_reward_breakdown = breakdown

        if (
            self.cfg.debug_print
            and self.cfg.debug_print_reward_every > 0
            and self._global_step > 0
            and self._global_step % self.cfg.debug_print_reward_every == 0
        ):
            print(
                f"\n--- REWARD DEBUG (step {self._global_step}) ---\n"
                f"  pos         : {breakdown['pos']:+.4f}\n"
                f"  action      : {breakdown['action']:+.4f}\n"
                f"  smooth      : {breakdown['smooth']:+.4f}\n"
                f"  joint_speed : {breakdown['joint_speed']:+.4f}\n"
                f"  joint_limit : {breakdown['joint_limit']:+.4f}\n"
                f"  wall        : {breakdown['wall']:+.4f}\n"
                f"  ik          : {breakdown['ik']:+.4f}\n"
                f"  success     : {breakdown['success']:+.4f}\n"
                f"  total       : {breakdown['total']:+.4f}\n"
                f"  error_norm  : {1000*breakdown['error_norm_m']:.2f} mm"
            )

        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        self.extras.setdefault("log", {}).update({
            "debug/error_norm_m": breakdown["error_norm_m"],
            "debug/pos_reward": breakdown["pos"],
            "debug/close_bonus": breakdown["close"],
            "debug/action_penalty": breakdown["action"],
            "debug/smooth_penalty": breakdown["smooth"],
            "debug/joint_speed_penalty": breakdown["joint_speed"],
            "debug/joint_limit_penalty": breakdown["joint_limit"],
            "debug/wall_penalty": breakdown["wall"],
            "debug/ik_penalty": breakdown["ik"],
            "debug/success_bonus": breakdown["success"],
            "debug/total_reward": breakdown["total"],
        })

        if self.cfg.debug_csv and self._csv_logger.enabled:
            tip0 = tip_pos_w[0].detach().cpu().tolist()
            tgt0 = self.target_pos_w[0].detach().cpu().tolist()
            err0 = float(pos_norm[0].item())
            rew0 = float(reward[0].item())
            act0 = self.actions[0].detach().cpu().tolist()
            q0 = q[0].detach().cpu().tolist()
            dq0 = dq[0].detach().cpu().tolist()
            sc0 = int(self.success_counter[0].item())
            self._csv_logger.append(
                step=self._episode_step,
                tip_xyz=tip0,
                target_xyz=tgt0,
                error_norm=err0,
                reward=rew0,
                action_xyz=act0,
                q=q0,
                dq=dq0,
                success_counter=sc0,
            )
            self._episode_rewards.append(rew0)
            self._episode_errors.append(err0)

        self._global_step += 1
        self._episode_step += 1
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._initialize_robot_handles_and_buffers()
        tip_pos_w = self.robot.data.body_pos_w[:, self.tip_body_id[0], :]
        wall_cross = tip_pos_w[:, 1] < (self.socket_pos_w[:, 1] - self.cfg.wall_margin)
        success = self.success_counter >= self.cfg.success_hold_steps
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        terminated = success | (wall_cross & (self.episode_length_buf > 5)) | (~self.ik_valid)
        truncated = timeout
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_list = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
        env0_resetting = 0 in env_ids_list
        if env0_resetting and self._episode_step > 0:
            self._finalize_episode()

        super()._reset_idx(env_ids)
        self._initialize_robot_handles_and_buffers()

        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        socket_x = torch.empty(n, device=self.device).uniform_(*self.cfg.socket_x_range)
        socket_y = torch.empty(n, device=self.device).uniform_(*self.cfg.socket_y_range)
        socket_z = torch.empty(n, device=self.device).uniform_(*self.cfg.socket_z_range)
        socket_pos_local = torch.stack([socket_x, socket_y, socket_z], dim=-1)
        target_pos_local = socket_pos_local.clone()
        target_pos_local[:, 1] += self.cfg.preinsert_distance

        env_origins = self.scene.env_origins[env_ids_t]
        self.socket_pos_w[env_ids_t] = socket_pos_local + env_origins
        self.target_pos_w[env_ids_t] = target_pos_local + env_origins

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        reset_mode = self._reset_mode()
        if reset_mode == "analytic":
            start_pos_local = target_pos_local.clone()
            start_pos_local[:, 1] += self.cfg.start_offset_y
            q_start, ik_ok = analytic_ik_position(start_pos_local)
            q_start = torch.where(ik_ok.unsqueeze(-1), q_start, torch.zeros_like(q_start))
            joint_pos[:, self.arm_joint_ids] = q_start
        else:
            # Safest model-validation reset: use the USD/default zero joint pose.
            joint_pos[:, self.arm_joint_ids] = 0.0

        if self.passive_joint_ids:
            for jid in self.passive_joint_ids:
                joint_pos[:, jid] = 0.0

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Also set the drive targets to the reset pose so the arm does not jump at the first step.
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # Store the pose used by MAXARM_DEBUG_MODE=hold. This makes hold mode a true
        # freeze test instead of a drive-strength test.
        self._hold_joint_pos_all = self.robot.data.joint_pos.clone()

        self.actions[env_ids_t] = 0.0
        self.prev_actions[env_ids_t] = 0.0
        self.q_target_all[env_ids_t] = joint_pos
        self.q_target[env_ids_t] = joint_pos[:, self.arm_joint_ids]
        self.success_counter[env_ids_t] = 0
        self.ik_valid[env_ids_t] = True

    # ------------------------------------------------------------------
    # Episode finalization
    # ------------------------------------------------------------------
    def _finalize_episode(self) -> None:
        csv_path: str | None = None
        if self.cfg.debug_csv and self._csv_logger.enabled:
            csv_path = self._csv_logger.save(self._episode_id)

        if self.cfg.debug_print:
            ep_len = self._episode_step
            errors = self._episode_errors
            rewards = self._episode_rewards
            final_err = errors[-1] if errors else float("nan")
            min_err = min(errors) if errors else float("nan")
            mean_rew = (sum(rewards) / len(rewards)) if rewards else float("nan")
            sc = int(self.success_counter[0].item()) if self.success_counter is not None else 0
            stable = sc >= self.cfg.success_hold_steps
            print(
                f"\n===== EPISODE SUMMARY =====\n"
                f"  episode id           : {self._episode_id}\n"
                f"  episode length steps : {ep_len}\n"
                f"  final error          : {1000*final_err:.2f} mm\n"
                f"  min error            : {1000*min_err:.2f} mm\n"
                f"  mean reward          : {mean_rew:+.4f}\n"
                f"  stable success?      : {stable}\n"
                f"  success counter      : {sc}/{self.cfg.success_hold_steps}\n"
                f"  csv saved to         : {csv_path if csv_path else '(disabled or empty)'}\n"
                f"==========================="
            )

        self._episode_id += 1
        self._episode_step = 0
        self._episode_rewards = []
        self._episode_errors = []
