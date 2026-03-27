import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EduCodeEnv

MODELS_DIR_PPO      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pg", "ppo")
MODELS_DIR_A2C      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pg", "a2c")
MODELS_DIR_REINFORCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pg", "reinforce")
for d in [MODELS_DIR_PPO, MODELS_DIR_A2C, MODELS_DIR_REINFORCE]:
    os.makedirs(d, exist_ok=True)


class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._ep_reward      = 0.0
        self._ep_len         = 0

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


# ---------------------------------------------------------------------------
# REINFORCE (vanilla policy gradient) implemented from scratch with PyTorch
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class ReinforceAgent:
    """
    Vanilla REINFORCE with optional entropy regularisation.
    """
    def __init__(self, obs_dim, act_dim, learning_rate=1e-3,
                 gamma=0.99, entropy_coef=0.01, hidden=64):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.policy       = PolicyNetwork(obs_dim, act_dim, hidden)
        self.optimiser    = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.log_probs    = []
        self.rewards      = []
        self.entropies    = []

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.policy(obs_t)
        dist  = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        return action.item()

    def update(self):
        R       = 0.0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss  = []
        entropy_loss = []
        for log_prob, ret, ent in zip(self.log_probs, returns, self.entropies):
            policy_loss.append(-log_prob * ret)
            entropy_loss.append(-ent)

        loss = torch.stack(policy_loss).sum() + \
               self.entropy_coef * torch.stack(entropy_loss).sum()

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.log_probs = []
        self.rewards   = []
        self.entropies = []
        return loss.item()


def train_reinforce(
    run_id,
    learning_rate=1e-3,
    gamma=0.99,
    entropy_coef=0.01,
    hidden_size=64,
    n_episodes=800,
    student_profile="average",
):
    env = EduCodeEnv(student_profile=student_profile)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = ReinforceAgent(
        obs_dim       = obs_dim,
        act_dim       = act_dim,
        learning_rate = learning_rate,
        gamma         = gamma,
        entropy_coef  = entropy_coef,
        hidden        = hidden_size,
    )

    episode_rewards  = []
    episode_lengths  = []
    entropy_history  = []

    print(f"\nStarting REINFORCE Run {run_id} | lr={learning_rate} gamma={gamma} "
          f"entropy={entropy_coef} hidden={hidden_size}\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps        = 0
        done         = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            steps        += 1
            done = terminated or truncated

        loss = agent.update()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        entropy_history.append(
            float(np.mean([e.item() for e in agent.entropies])) if agent.entropies else 0.0
        )

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep+1:>4} | avg reward (last 100): {avg:.2f}")

    env.close()

    save_path = os.path.join(MODELS_DIR_REINFORCE, f"reinforce_run_{run_id}.pt")
    torch.save(agent.policy.state_dict(), save_path)
    print(f"REINFORCE model saved to {save_path}")

    return agent, episode_rewards, episode_lengths, entropy_history


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(
    run_id,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    clip_range=0.2,
    gae_lambda=0.95,
    total_timesteps=40000,
    student_profile="average",
    verbose=0,
):
    env      = make_env(profile=student_profile)
    eval_env = make_env(profile=student_profile)

    model = PPO(
        policy       = "MlpPolicy",
        env          = env,
        learning_rate = learning_rate,
        gamma        = gamma,
        n_steps      = n_steps,
        batch_size   = batch_size,
        n_epochs     = n_epochs,
        ent_coef     = ent_coef,
        clip_range   = clip_range,
        gae_lambda   = gae_lambda,
        verbose      = verbose,
        device       = "auto",
    )

    reward_logger = RewardLogger()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODELS_DIR_PPO,
        log_path             = MODELS_DIR_PPO,
        eval_freq            = 5000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
    )

    print(f"\nStarting PPO Run {run_id} | lr={learning_rate} gamma={gamma} "
          f"n_steps={n_steps} batch={batch_size} ent_coef={ent_coef} clip={clip_range}\n")

    model.learn(
        total_timesteps = total_timesteps,
        callback        = [reward_logger, eval_callback],
        progress_bar    = True,
    )

    save_path = os.path.join(MODELS_DIR_PPO, f"ppo_run_{run_id}")
    model.save(save_path)
    print(f"PPO model saved to {save_path}")

    env.close()
    eval_env.close()
    return model, reward_logger


# ---------------------------------------------------------------------------
# A2C training
# ---------------------------------------------------------------------------

def train_a2c(
    run_id,
    learning_rate=7e-4,
    gamma=0.99,
    n_steps=5,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gae_lambda=1.0,
    total_timesteps=40000,
    student_profile="average",
    verbose=0,
):
    env      = make_env(profile=student_profile)
    eval_env = make_env(profile=student_profile)

    model = A2C(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = learning_rate,
        gamma         = gamma,
        n_steps       = n_steps,
        ent_coef      = ent_coef,
        vf_coef       = vf_coef,
        max_grad_norm = max_grad_norm,
        gae_lambda    = gae_lambda,
        verbose       = verbose,
        device        = "auto",
    )

    reward_logger = RewardLogger()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODELS_DIR_A2C,
        log_path             = MODELS_DIR_A2C,
        eval_freq            = 5000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
    )

    print(f"\nStarting A2C Run {run_id} | lr={learning_rate} gamma={gamma} "
          f"n_steps={n_steps} ent={ent_coef} vf={vf_coef}\n")

    model.learn(
        total_timesteps = total_timesteps,
        callback        = [reward_logger, eval_callback],
        progress_bar    = True,
    )

    save_path = os.path.join(MODELS_DIR_A2C, f"a2c_run_{run_id}")
    model.save(save_path)
    print(f"A2C model saved to {save_path}")

    env.close()
    eval_env.close()
    return model, reward_logger


# ---------------------------------------------------------------------------
# Hyperparameter grids for the 10 required runs per algorithm
# ---------------------------------------------------------------------------

PPO_RUNS = [
    dict(run_id=1,  learning_rate=3e-4, gamma=0.99, n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=2,  learning_rate=1e-4, gamma=0.99, n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=3,  learning_rate=3e-4, gamma=0.95, n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=4,  learning_rate=3e-4, gamma=0.99, n_steps=256,  batch_size=64,  n_epochs=10, ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=5,  learning_rate=3e-4, gamma=0.99, n_steps=512,  batch_size=128, n_epochs=10, ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=6,  learning_rate=3e-4, gamma=0.99, n_steps=512,  batch_size=64,  n_epochs=5,  ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=7,  learning_rate=3e-4, gamma=0.99, n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.05,  clip_range=0.2,  gae_lambda=0.95),
    dict(run_id=8,  learning_rate=3e-4, gamma=0.99, n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01,  clip_range=0.3,  gae_lambda=0.95),
    dict(run_id=9,  learning_rate=3e-4, gamma=0.99, n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01,  clip_range=0.2,  gae_lambda=0.80),
    dict(run_id=10, learning_rate=5e-4, gamma=0.97, n_steps=1024, batch_size=128, n_epochs=15, ent_coef=0.02,  clip_range=0.15, gae_lambda=0.90),
]

A2C_RUNS = [
    dict(run_id=1,  learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(run_id=2,  learning_rate=3e-4, gamma=0.99, n_steps=5,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(run_id=3,  learning_rate=7e-4, gamma=0.95, n_steps=5,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(run_id=4,  learning_rate=7e-4, gamma=0.99, n_steps=10, ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(run_id=5,  learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.05,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(run_id=6,  learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.01,  vf_coef=0.25, max_grad_norm=0.5, gae_lambda=1.0),
    dict(run_id=7,  learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.3, gae_lambda=1.0),
    dict(run_id=8,  learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=0.9),
    dict(run_id=9,  learning_rate=1e-3, gamma=0.99, n_steps=20, ent_coef=0.02,  vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=0.95),
    dict(run_id=10, learning_rate=5e-4, gamma=0.97, n_steps=15, ent_coef=0.03,  vf_coef=0.4,  max_grad_norm=0.4, gae_lambda=0.92),
]

REINFORCE_RUNS = [
    dict(run_id=1,  learning_rate=1e-3, gamma=0.99, entropy_coef=0.01, hidden_size=64),
    dict(run_id=2,  learning_rate=5e-4, gamma=0.99, entropy_coef=0.01, hidden_size=64),
    dict(run_id=3,  learning_rate=1e-3, gamma=0.95, entropy_coef=0.01, hidden_size=64),
    dict(run_id=4,  learning_rate=1e-3, gamma=0.99, entropy_coef=0.05, hidden_size=64),
    dict(run_id=5,  learning_rate=1e-3, gamma=0.99, entropy_coef=0.01, hidden_size=128),
    dict(run_id=6,  learning_rate=2e-3, gamma=0.99, entropy_coef=0.01, hidden_size=64),
    dict(run_id=7,  learning_rate=1e-3, gamma=0.99, entropy_coef=0.10, hidden_size=64),
    dict(run_id=8,  learning_rate=1e-3, gamma=0.90, entropy_coef=0.01, hidden_size=64),
    dict(run_id=9,  learning_rate=1e-3, gamma=0.99, entropy_coef=0.00, hidden_size=32),
    dict(run_id=10, learning_rate=3e-4, gamma=0.97, entropy_coef=0.02, hidden_size=128),
]


def plot_pg_results(all_rewards, all_labels, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    ax1, ax2 = axes

    for rewards, label in zip(all_rewards, all_labels):
        if len(rewards) < 10:
            continue
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax1.plot(smoothed, label=label, alpha=0.8)

    ax1.set_title("Cumulative Episode Reward (smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    final_means = [np.mean(r[-50:]) if len(r) >= 50 else np.mean(r) for r in all_rewards]
    bar_labels  = [l.split("|")[0].strip() for l in all_labels]
    colours     = ["#27ae60" if m > 0 else "#e74c3c" for m in final_means]
    ax2.barh(bar_labels, final_means, color=colours, alpha=0.85)
    ax2.axvline(0, color="white", linewidth=0.8)
    ax2.set_title("Mean Reward (last 50 episodes)")
    ax2.set_xlabel("Mean Reward")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train policy gradient agents on EduCode")
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce", "all"], default="all")
    args = parser.parse_args()

    if args.algo in ("reinforce", "all"):
        print("\n========== REINFORCE ==========")
        reinforce_rewards = []
        reinforce_labels  = []
        for cfg in REINFORCE_RUNS:
            run_cfg = {k: v for k, v in cfg.items()}
            run_id  = run_cfg.pop("run_id")
            _, rewards, _, _ = train_reinforce(run_id=run_id, **run_cfg)
            reinforce_rewards.append(rewards)
            reinforce_labels.append(f"Run {run_id} lr={run_cfg['learning_rate']}")

        plot_pg_results(
            reinforce_rewards, reinforce_labels,
            "REINFORCE Training - EduCode Rwanda",
            save_path=os.path.join(MODELS_DIR_REINFORCE, "reinforce_curves.png"),
        )

    if args.algo in ("ppo", "all"):
        print("\n========== PPO ==========")
        ppo_rewards = []
        ppo_labels  = []
        for cfg in PPO_RUNS:
            run_cfg = {k: v for k, v in cfg.items()}
            run_id  = run_cfg.pop("run_id")
            model, logger = train_ppo(run_id=run_id, **run_cfg)
            ppo_rewards.append(logger.episode_rewards)
            ppo_labels.append(f"Run {run_id} lr={run_cfg['learning_rate']}")

        plot_pg_results(
            ppo_rewards, ppo_labels,
            "PPO Training - EduCode Rwanda",
            save_path=os.path.join(MODELS_DIR_PPO, "ppo_curves.png"),
        )

    if args.algo in ("a2c", "all"):
        print("\n========== A2C ==========")
        a2c_rewards = []
        a2c_labels  = []
        for cfg in A2C_RUNS:
            run_cfg = {k: v for k, v in cfg.items()}
            run_id  = run_cfg.pop("run_id")
            model, logger = train_a2c(run_id=run_id, **run_cfg)
            a2c_rewards.append(logger.episode_rewards)
            a2c_labels.append(f"Run {run_id} lr={run_cfg['learning_rate']}")

        plot_pg_results(
            a2c_rewards, a2c_labels,
            "A2C Training - EduCode Rwanda",
            save_path=os.path.join(MODELS_DIR_A2C, "a2c_curves.png"),
        )

    print("\nAll policy gradient training runs finished.")
