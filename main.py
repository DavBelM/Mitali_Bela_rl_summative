import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import EduCodeEnv, TOPICS, ACTIONS


def load_sb3_model(algo, model_path):
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def run_random_agent(n_episodes=3, render=True, profile="average"):
    """
    Run an agent that picks actions at random.
    Used to demonstrate the environment and visualisation before any training.
    """
    env = EduCodeEnv(render_mode="human" if render else None, student_profile=profile)

    print("\nRunning random agent (no model, pure environment demo)...")
    print(f"Student profile: {profile}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print("-" * 60)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step         = 0
        done         = False

        print(f"\nEpisode {ep + 1}")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step         += 1
            done          = terminated or truncated

            if render:
                env.render()
                time.sleep(0.05)

            print(
                f"  step {step:>3} | {ACTIONS[action]:<22} | "
                f"topic: {info['topic']:<25} | "
                f"mastery: {info['mastery']:.2f} | "
                f"engagement: {info['engagement']:.2f} | "
                f"reward: {reward:+.2f}"
            )

        print(f"Episode {ep + 1} finished | total reward: {total_reward:.2f} | steps: {step}")

    env.close()


def run_trained_agent(algo, model_path, n_episodes=5, render=True, profile="average"):
    """
    Load a saved model and run it in the environment.
    This is the main showcase function for the best performing agent.
    """
    print(f"\nLoading {algo.upper()} model from: {model_path}")
    model = load_sb3_model(algo, model_path)

    env = EduCodeEnv(render_mode="human" if render else None, student_profile=profile)

    print(f"\nRunning trained {algo.upper()} agent | profile: {profile}")
    print("=" * 60)

    all_rewards   = []
    all_steps     = []
    topics_totals = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward   = 0.0
        step           = 0
        done           = False
        topics_covered = set()

        print(f"\nEpisode {ep + 1} / {n_episodes}")
        print("-" * 60)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            step         += 1
            done          = terminated or truncated
            topics_covered.add(info["topic"])

            if render:
                env.render()
                time.sleep(0.06)

            print(
                f"  step {step:>3} | {info['action_name']:<22} | "
                f"topic: {info['topic']:<25} | "
                f"mastery: {info['mastery']:.2f} | "
                f"eng: {info['engagement']:.2f} | "
                f"done_topics: {info['topics_completed']} | "
                f"r: {reward:+.2f}"
            )

        all_rewards.append(total_reward)
        all_steps.append(step)
        topics_totals.append(info["topics_completed"])

        print(f"\nEpisode {ep + 1} summary:")
        print(f"  Total reward    : {total_reward:.2f}")
        print(f"  Steps taken     : {step}")
        print(f"  Topics mastered : {info['topics_completed']} / {len(TOPICS)}")
        print(f"  Topics visited  : {', '.join(sorted(topics_covered))}")

    env.close()

    print("\n" + "=" * 60)
    print(f"Results over {n_episodes} episodes:")
    print(f"  Mean reward    : {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean steps     : {np.mean(all_steps):.1f}")
    print(f"  Mean topics    : {np.mean(topics_totals):.1f} / {len(TOPICS)}")
    print("=" * 60)


def find_best_model(algo):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    candidates = {
        "dqn":      os.path.join(base, "dqn", "best_model"),
        "ppo":      os.path.join(base, "pg", "ppo", "best_model"),
        "a2c":      os.path.join(base, "pg", "a2c", "best_model"),
    }
    path = candidates.get(algo)
    if path and (os.path.exists(path + ".zip") or os.path.exists(path)):
        return path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EduCode Rwanda - RL Tutor Agent Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["random", "trained"],
        default="trained",
        help="Run random agent (env demo) or trained model",
    )
    parser.add_argument(
        "--algo",
        choices=["dqn", "ppo", "a2c"],
        default="ppo",
        help="Which algorithm model to load (used in trained mode)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model file (without .zip). Auto-detected if not given.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--profile",
        choices=["struggling", "average", "advanced"],
        default="average",
        help="Simulated student profile",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable pygame visualisation (terminal output only)",
    )

    args = parser.parse_args()

    render = not args.no_render

    if args.mode == "random":
        run_random_agent(
            n_episodes = args.episodes,
            render     = render,
            profile    = args.profile,
        )
    else:
        model_path = args.model_path
        if model_path is None:
            model_path = find_best_model(args.algo)
        if model_path is None:
            print(f"No saved model found for {args.algo}.")
            print("Train first with:  python training/dqn_training.py")
            print("                   python training/pg_training.py --algo ppo")
            sys.exit(1)

        run_trained_agent(
            algo       = args.algo,
            model_path = model_path,
            n_episodes = args.episodes,
            render     = render,
            profile    = args.profile,
        )
