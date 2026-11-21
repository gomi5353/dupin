"""
References: https://github.com/LightwheelAI/leisaac
Unified data generation script using state machines.

Selects the state machine for --task and runs --num_demos episodes. The task's
reset events randomize the object positions at the start of every episode.

With --num_envs N, the N envs run one episode each per round, in lockstep
(the state machine steps all envs together and the round ends when it is
done). Each env's episode is judged and recorded on its own: a round records
the envs that succeeded. --num_demos counts episodes over all envs and is
rounded up to whole rounds.

Usage:
    python scripts/datagen/generate.py \
        --task HCIS-CupStacking-SingleArm-v0 --num_demos 10 \
        --num_envs 1 --device cuda --enable_cameras \
        --record --use_lerobot_recorder --lerobot_dataset_repo_id <hf_user>/cup_stacking
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of episodes to run (successful or not).",
)
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=60, help="Lerobot Dataset frames per second (one frame per env.step at 60 Hz).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import simulator.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager

from simulator.datagen.state_machine.cup_stacking import CupStackingStateMachine
from simulator.datagen.state_machine.cutlery_arrangement import CutleryArrangementStateMachine
from simulator.datagen.state_machine.toy_blocks_collection import ToyBlocksCollectionStateMachine

# Maps gym task id → (StateMachineClass, device_type)
TASK_REGISTRY = {
    "HCIS-CupStacking-SingleArm-v0": (CupStackingStateMachine, "keyboard"),
    "HCIS-ToyBlocksCollection-SingleArm-v0": (ToyBlocksCollectionStateMachine, "keyboard"),
    "HCIS-CutleryArrangement-SingleArm-v0": (CutleryArrangementStateMachine, "keyboard"),
}

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool | torch.Tensor):
    """Set the success term the recorder reads at the next reset: one bool for every env, or a (num_envs,) mask."""
    if isinstance(success, torch.Tensor):
        mask = success.to(device=env.device, dtype=torch.bool).clone()
    else:
        mask = torch.full((env.num_envs,), bool(success), dtype=torch.bool, device=env.device)
    if hasattr(env, "termination_manager"):
        if "success" not in env.termination_manager.active_terms:
            return  # not recording: success term was removed in the env cfg
        env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: mask))
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = bool(mask.all().item())


def _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name):
    """Configure termination and recorder settings on env_cfg."""
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


def _replace_recorder_manager(env, env_cfg, args_cli):
    """Replace the default recorder manager with streaming or lerobot recorder."""
    del env.recorder_manager
    if args_cli.use_lerobot_recorder:
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from simulator.utils.lerobot_recorder import LeRobotRecorderManager

        dataset_cfg = LeRobotDatasetCfg(
            repo_id=args_cli.lerobot_dataset_repo_id,
            fps=args_cli.lerobot_dataset_fps,
        )
        env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
    else:
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


# z below which a task object is considered to have fallen off the table.
# Objects sit at object_z ≈ 0.05; anything under the table surface trips this.
_FALL_THRESHOLD_Z: float = 0.0


def _any_object_fell(env, object_names, z_threshold: float) -> bool:
    """Return True if any named scene object has root_pos_w.z below z_threshold."""
    for name in object_names:
        try:
            obj = env.scene[name]
        except KeyError:
            continue
        if torch.any(obj.data.root_pos_w[:, 2] < z_threshold).item():
            return True
    return False


def _on_episode_done(
    env,
    sm,
    args_cli,
    episodes,
    next_episode_idx,
    resume_recorded_demo_count,
    current_recorded_demo_count,
    start_record_state,
):
    """Handle end-of-episode logic.

    Returns (next_episode_idx, current_recorded_demo_count, start_record_state, should_break,
    success per env as a list of bools).
    """
    total_episodes = len(episodes)

    try:
        success_mask = sm.success_mask(env).bool()
    except Exception as e:
        print("Success check failed:", e)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    successes = success_mask.tolist()
    success = any(successes)

    if env.num_envs == 1:
        print("Episode success!" if success else "Episode failed!")
    else:
        print(f"Round done: {sum(successes)}/{env.num_envs} envs succeeded "
              f"(envs {[i for i, ok in enumerate(successes) if ok]}).")

    if start_record_state:
        if args_cli.record:
            print("Stop Recording!!!")
        start_record_state = False

    if args_cli.record and success:
        auto_terminate(env, success_mask)
        current_recorded_demo_count += sum(successes)
    else:
        auto_terminate(env, False)

    if (
        args_cli.record
        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        > current_recorded_demo_count
    ):
        current_recorded_demo_count = (
            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        )
        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

    # RecorderManager exports the episode from its pre-reset hook.  In
    # particular, LeRobotRecorderManager.finalize() only finalizes the dataset
    # writer; it does not export an episode that never reached reset.  Always
    # reset here, including after the final attempt, so the final successful
    # episode is flushed to parquet/video before leaving the loop.
    env.reset()
    sm.reset()
    auto_terminate(env, False)

    # This round ran episodes next_episode_idx .. next_episode_idx + num_envs - 1.
    if next_episode_idx + env.num_envs - 1 >= total_episodes:
        print(f"Completed all {total_episodes} episodes. Exiting the app.")
        return next_episode_idx, current_recorded_demo_count, start_record_state, True, successes

    next_episode_idx += env.num_envs

    return next_episode_idx, current_recorded_demo_count, start_record_state, False, successes


def main():
    """Run a state machine in a LeIsaac manipulation environment."""
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\nAvailable tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device = TASK_REGISTRY[task_name]

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    # Same action space, gravity and gripper settings as scripts/rollout.py.
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    if args_cli.num_demos <= 0:
        raise ValueError("--num_demos must be greater than 0.")
    episodes = list(range(args_cli.num_demos))

    is_direct_env = "Direct" in task_name
    _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    if args_cli.record:
        _replace_recorder_manager(env, env_cfg, args_cli)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    # one-time state machine setup (e.g. FK calibration)
    sm = SMClass()
    sm.setup(env)
    env.reset()
    sm.reset()

    fall_check_object_names = tuple(getattr(sm, "task_object_names", ()))

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    next_episode_idx = min(resume_recorded_demo_count, len(episodes))
    if next_episode_idx >= len(episodes):
        print(f"Resume count {next_episode_idx} ≥ total episodes {len(episodes)}; nothing to do.")
        env.close()
        simulation_app.close()
        return
    next_episode_idx += 1

    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    cnt = 1
    success_ID = []
    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if sm.is_episode_done:
                    (
                        next_episode_idx,
                        current_recorded_demo_count,
                        start_record_state,
                        should_break,
                        successes,
                    ) = _on_episode_done(
                        env,
                        sm,
                        args_cli,
                        episodes,
                        next_episode_idx,
                        resume_recorded_demo_count,
                        current_recorded_demo_count,
                        start_record_state,
                    )
                    for success in successes:
                        if success:
                            print(f"\033[92m[Data Usage]{cnt}/{len(episodes)} success.\033[0m")
                            success_ID.append(cnt)
                        else:
                            print(f"\033[91m[Data Usage]{cnt}/{len(episodes)} fail.\033[0m")
                        cnt += 1
                    if should_break:
                        break
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    env.step(actions)
                    sm.advance()

                    if fall_check_object_names and _any_object_fell(
                        env, fall_check_object_names, _FALL_THRESHOLD_Z
                    ):
                        print(
                            "[INFO] Task object fell off the table; aborting this "
                            "episode and skipping to next."
                        )
                        sm._episode_done = True

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()

    print(f"Successful episodes: {success_ID}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        # Isaac Sim can hang on shutdown after an error; exit hard so the shell sees the failure.
        os._exit(1)
