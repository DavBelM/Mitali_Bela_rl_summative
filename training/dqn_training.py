import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EduCodeEnv

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dqn")
os.makedirs(MODELS_DIR, exist_ok=True)


class RewardLogger(BaseCallback):
    """
    Collects episode rewards during training so we can plot them later.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.episode_lengths  = []
        self._ep_reward       = 0.0
        self._ep_len          = 0

    def _on_step(self):
        self._ep_reward += self.locals["rewards"][0]
        self._ep_len    += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self._ep_reward = 0.0
            self._ep_len    = 0
        return True


def make_env(profile="average"):
    env = EduCodeEnv(student_profile=profile)
    return Monitor(env)


def train_dqn(
    run_id,
    learning_rate=1e-3,
    gamma=0.99,
    batch_size=64,
    buffer_size=10000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    target_update_interval=500,
    train_freq=4,
    total_timesteps=40000,
    student_profile="average",
    verbose=0,
):
    """
    Train a DQN agent on the EduCode environment.

    Parameters match those expected in the hyperparameter comparison table.
    Returns the trained model and the reward logger callback.
    """
    env      = make_env(profile=student_profile)
    eval_env = make_env(profile=student_profile)

    model = DQN(
        policy                  = "MlpPolicy",
        env                     = env,
        learning_rate           = learning_rate,
        gamma                   = gamma,
        batch_size              = batch_size,
        buffer_size             = buffer_size,
        exploration_fraction    = exploration_fraction,
        exploration_final_eps   = exploration_final_eps,
        target_update_interval  = target_update_interval,
        train_freq              = train_freq,
        verbose                 = verbose,
        device                  = "auto",
    )

    reward_logger = RewardLogger()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODELS_DIR,
        log_path             = MODELS_DIR,
        eval_freq            = 5000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
    )

    print(f"\nStarting DQN Run {run_id} | lr={learning_rate} gamma={gamma} "
          f"batch={batch_size} buf={buffer_size} "
          f"eps_frac={exploration_fraction} eps_final={exploration_final_eps}\n")

    model.learn(
        total_timesteps = total_timesteps,
        callback        = [reward_logger, eval_callback],
        progress_bar    = True,
    )

    save_path = os.path.join(MODELS_DIR, f"dqn_run_{run_id}")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    env.close()
    eval_env.close()
    return model, reward_logger


def evaluate_model(model, n_episodes=20, student_profile="average"):
    env = make_env(profile=student_profile)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )
    env.close()
    return mean_reward, std_reward


def plot_training_results(reward_loggers, run_labels, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DQN Training Results - EduCode Rwanda", fontsize=14)

    ax1 = axes[0]
    for logger, label in zip(reward_loggers, run_labels):
        rewards = logger.episode_rewards
        if len(rewards) == 0:
            continue
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax1.plot(smoothed, label=label, alpha=0.85)
    ax1.set_title("Cumulative Episode Reward (smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for logger, label in zip(reward_loggers, run_labels):
        lengths = logger.episode_lengths
        if len(lengths) == 0:
            continue
        smoothed = np.convolve(lengths, np.ones(10) / 10, mode="valid")
        ax2.plot(smoothed, label=label, alpha=0.85)
    ax2.set_title("Episode Length (smoothed)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()


# Hyperparameter configurations for the 10 required runs
HYPERPARAM_RUNS = [
    dict(run_id=1,  learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=500,  train_freq=4),
    dict(run_id=2,  learning_rate=5e-4,  gamma=0.99, batch_size=64,  buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=500,  train_freq=4),
    dict(run_id=3,  learning_rate=1e-3,  gamma=0.95, batch_size=64,  buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=500,  train_freq=4),
    dict(run_id=4,  learning_rate=1e-3,  gamma=0.99, batch_size=128, buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=500,  train_freq=4),
    dict(run_id=5,  learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=50000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=500,  train_freq=4),
    dict(run_id=6,  learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=10000, exploration_fraction=0.35, exploration_final_eps=0.05, target_update_interval=500,  train_freq=4),
    dict(run_id=7,  learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.10, target_update_interval=500,  train_freq=4),
    dict(run_id=8,  learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=1000, train_freq=4),
    dict(run_id=9,  learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=10000, exploration_fraction=0.20, exploration_final_eps=0.05, target_update_interval=500,  train_freq=1),
    dict(run_id=10, learning_rate=2e-3,  gamma=0.95, batch_size=128, buffer_size=50000, exploration_fraction=0.30, exploration_final_eps=0.02, target_update_interval=250,  train_freq=2),
]


if __name__ == "__main__":
    loggers     = []
    labels      = []
    final_scores = []

    for cfg in HYPERPARAM_RUNS:
        run_cfg = {k: v for k, v in cfg.items()}
        run_id  = run_cfg.pop("run_id")

        model, logger = train_dqn(run_id=run_id, **run_cfg)
        mean_r, std_r = evaluate_model(model)
        final_scores.append((run_id, mean_r, std_r))

        loggers.append(logger)
        labels.append(f"Run {run_id} lr={run_cfg['learning_rate']}")

        print(f"Run {run_id}: mean_reward={mean_r:.2f} +/- {std_r:.2f}")

    print("\nAll DQN runs complete.")
    print(f"{'Run':>4}  {'Mean Reward':>12}  {'Std':>8}")
    for run_id, mean_r, std_r in final_scores:
        print(f"{run_id:>4}  {mean_r:>12.2f}  {std_r:>8.2f}")

    plot_path = os.path.join(MODELS_DIR, "dqn_training_curves.png")
    plot_training_results(loggers, labels, save_path=plot_path)
