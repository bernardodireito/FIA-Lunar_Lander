import argparse
import importlib.util
import os
from collections import Counter

import gymnasium as gym
import numpy as np


def load_project():
    spec = importlib.util.spec_from_file_location("lander", "NE-LunarLander-alunos.py")
    lander = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lander)
    return lander


def load_controller(lander, filename, mode):
    bests = lander.load_bests(filename)
    if mode == "last":
        chosen = bests[-1]
    else:
        chosen = max(bests, key=lambda x: x[0])
    train_fitness, shape, genotype = chosen
    return float(train_fitness), shape, genotype


def landing_checks(lander, observation):
    x = float(observation[0])
    vy = float(observation[3])
    theta = float(observation[4])
    contact_left = int(observation[6])
    contact_right = int(observation[7])

    return {
        "success": lander.check_successful_landing(observation),
        "legs_touching": contact_left == 1 and contact_right == 1,
        "on_pad": abs(x) <= 0.2,
        "slow_vy": vy > -0.2,
        "upright": abs(theta) < np.deg2rad(20),
    }


def failure_reasons(lander, observation):
    checks = landing_checks(lander, observation)
    if checks["success"]:
        return ["success"]

    reasons = []
    if not checks["legs_touching"]:
        reasons.append("legs_not_touching")
    if not checks["on_pad"]:
        reasons.append("outside_pad")
    if not checks["slow_vy"]:
        reasons.append("falling_too_fast")
    if not checks["upright"]:
        reasons.append("bad_angle")
    return reasons


def simulate_with_trace(lander, genotype, shape, seed=None, render_mode=None):
    env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
        continuous=True,
        gravity=lander.GRAVITY,
        enable_wind=lander.ENABLE_WIND,
        wind_power=lander.WIND_POWER,
        turbulence_power=lander.TURBULENCE_POWER,
    )

    lander.SHAPE = shape
    observation, info = env.reset(seed=seed)
    observations = [observation]
    actions = []
    rewards = []
    terminated = False
    truncated = False

    for _ in range(lander.STEPS):
        action = lander.controller_action(shape, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    env.close()

    fitness, success = lander.objective_function(observations)
    return {
        "fitness": float(fitness),
        "success": bool(success),
        "terminated": terminated,
        "truncated": truncated,
        "steps": len(actions),
        "reward_sum": sum(rewards),
        "observations": observations,
        "actions": actions,
    }


def summarize_controller(lander, filename, mode, episodes, show_cases):
    train_fitness, shape, genotype = load_controller(lander, filename, mode)

    failures_second_last = Counter()
    failures_last = Counter()
    terminations = Counter()
    final_values = []
    action_values = []
    successes = 0
    total_fitness = 0.0
    total_reward = 0.0
    examples = []

    for seed in range(episodes):
        result = simulate_with_trace(lander, genotype, shape, seed=seed)
        observations = result["observations"]
        second_last = observations[-2]
        last = observations[-1]

        total_fitness += result["fitness"]
        total_reward += result["reward_sum"]
        successes += int(result["success"])

        for reason in failure_reasons(lander, second_last):
            failures_second_last[reason] += 1
        for reason in failure_reasons(lander, last):
            failures_last[reason] += 1

        if result["terminated"]:
            terminations["terminated"] += 1
        elif result["truncated"]:
            terminations["truncated"] += 1
        else:
            terminations["max_steps_without_done"] += 1

        final_values.append(last)
        action_values.extend(result["actions"])

        if not result["success"] and len(examples) < show_cases:
            examples.append((seed, result, second_last, last))

    final_values = np.array(final_values)
    action_values = np.array(action_values)

    print(f"\n{filename} ({mode})")
    print(f"  train_fitness_in_log: {train_fitness:.3f}")
    print(f"  test_avg_fitness:     {total_fitness / episodes:.3f}")
    print(f"  test_avg_reward:      {total_reward / episodes:.3f}")
    print(f"  success_rate:         {successes}/{episodes} ({successes / episodes:.1%})")
    print(f"  endings:              {dict(terminations)}")
    print(f"  fail reasons [-2]:    {dict(failures_second_last)}")
    print(f"  fail reasons [-1]:    {dict(failures_last)}")

    labels = ["x", "y", "vx", "vy", "theta", "vtheta", "left_leg", "right_leg"]
    print("  final observation means:")
    for i, label in enumerate(labels):
        print(f"    {label:9s} mean={final_values[:, i].mean():8.3f} min={final_values[:, i].min():8.3f} max={final_values[:, i].max():8.3f}")

    if len(action_values) > 0:
        print("  action means:")
        print(f"    main_engine mean={action_values[:, 0].mean():8.3f} min={action_values[:, 0].min():8.3f} max={action_values[:, 0].max():8.3f}")
        print(f"    side_engine mean={action_values[:, 1].mean():8.3f} min={action_values[:, 1].min():8.3f} max={action_values[:, 1].max():8.3f}")

    for seed, result, second_last, last in examples:
        print(f"  example seed={seed} steps={result['steps']} fitness={result['fitness']:.3f} reward={result['reward_sum']:.3f}")
        print(f"    reasons[-2]={failure_reasons(lander, second_last)}")
        print(f"    obs[-2]={np.array2string(np.array(second_last), precision=3, suppress_small=True)}")
        print(f"    reasons[-1]={failure_reasons(lander, last)}")
        print(f"    obs[-1]={np.array2string(np.array(last), precision=3, suppress_small=True)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Specific log file to debug. By default, uses all log*.txt files.")
    parser.add_argument("--mode", choices=["best", "last"], default="best")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--show-cases", type=int, default=3)
    args = parser.parse_args()

    lander = load_project()

    if args.log is not None:
        logs = [args.log]
    else:
        logs = sorted(
            f for f in os.listdir(".")
            if f.startswith("log") and f.endswith(".txt")
        )

    if not logs:
        raise SystemExit("No log files found.")

    for filename in logs:
        summarize_controller(lander, filename, args.mode, args.episodes, args.show_cases)


if __name__ == "__main__":
    main()
