import argparse
import csv
import importlib.util
import os
import re
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


def find_log_files(log_root):
    def collect_logs(search_root):
        experiment_logs = []
        all_logs = []
        if not os.path.exists(search_root):
            return experiment_logs, all_logs
        seen = set()
        for root, _, files in os.walk(search_root):
            for filename in files:
                if not (filename.startswith("log") and filename.endswith(".txt")):
                    continue
                path = os.path.join(root, filename)
                normalized_path = os.path.abspath(path)
                if normalized_path in seen:
                    continue
                seen.add(normalized_path)
                all_logs.append(path)
                if re.match(r"^log_exp[^_]+_run\d+\.txt$", filename):
                    experiment_logs.append(path)
        return experiment_logs, all_logs

    experiment_logs, all_logs = collect_logs(log_root)
    if not experiment_logs and not all_logs and log_root != ".":
        experiment_logs, all_logs = collect_logs(".")

    return sorted(experiment_logs if experiment_logs else all_logs)


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

    return {
        "filename": filename,
        "train_fitness": train_fitness,
        "test_avg_fitness": total_fitness / episodes,
        "test_avg_reward": total_reward / episodes,
        "success_rate": successes / episodes,
        "successes": successes,
        "episodes": episodes,
        "failures_last": dict(failures_last),
        "failures_second_last": dict(failures_second_last),
    }


def experiment_id_from_filename(filename):
    match = re.search(r"_exp([^_/\\]+)_run\d+\.txt$", filename)
    if match:
        return match.group(1)
    return "unknown"


def run_id_from_filename(filename):
    match = re.search(r"_run(\d+)\.txt$", filename)
    if match:
        return int(match.group(1))
    return -1


def load_training_curve(lander, filename):
    bests = lander.load_bests(filename)
    rows = []
    for generation, best in enumerate(bests):
        rows.append({
            "filename": filename,
            "experiment": experiment_id_from_filename(filename),
            "run": run_id_from_filename(filename),
            "generation": generation,
            "train_fitness": float(best[0]),
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_test_result(result):
    row = {
        "filename": result["filename"],
        "train_fitness": result["train_fitness"],
        "test_avg_fitness": result["test_avg_fitness"],
        "test_avg_reward": result["test_avg_reward"],
        "success_rate": result["success_rate"],
        "successes": result["successes"],
        "episodes": result["episodes"],
    }
    for reason in ["success", "legs_not_touching", "outside_pad", "falling_too_fast", "bad_angle"]:
        row[f"last_{reason}"] = result["failures_last"].get(reason, 0) / result["episodes"]
    return row


def summarize_results_by_experiment(results):
    grouped = {}
    for result in results:
        experiment_id = experiment_id_from_filename(result["filename"])
        grouped.setdefault(experiment_id, []).append(result)

    summary_rows = []
    for experiment_id in sorted(grouped.keys(), key=sort_experiment_key):
        experiment_results = grouped[experiment_id]
        success_rates = np.array([r["success_rate"] for r in experiment_results])
        fitness_values = np.array([r["test_avg_fitness"] for r in experiment_results])
        reward_values = np.array([r["test_avg_reward"] for r in experiment_results])
        legs_values = np.array([r["failures_last"].get("legs_not_touching", 0) / r["episodes"] for r in experiment_results])
        outside_values = np.array([r["failures_last"].get("outside_pad", 0) / r["episodes"] for r in experiment_results])
        fast_values = np.array([r["failures_last"].get("falling_too_fast", 0) / r["episodes"] for r in experiment_results])
        angle_values = np.array([r["failures_last"].get("bad_angle", 0) / r["episodes"] for r in experiment_results])
        best = max(experiment_results, key=lambda r: r["success_rate"])

        summary_rows.append({
            "experiment": experiment_id,
            "runs": len(experiment_results),
            "success_mean": float(success_rates.mean()),
            "success_std": float(success_rates.std()),
            "fitness_mean": float(fitness_values.mean()),
            "fitness_std": float(fitness_values.std()),
            "reward_mean": float(reward_values.mean()),
            "last_legs_not_touching_mean": float(legs_values.mean()),
            "last_outside_pad_mean": float(outside_values.mean()),
            "last_falling_too_fast_mean": float(fast_values.mean()),
            "last_bad_angle_mean": float(angle_values.mean()),
            "best_log": best["filename"],
        })
    return summary_rows


def write_experiment_summary_txt(path, summary_rows):
    if not summary_rows:
        return

    best_row = max(summary_rows, key=lambda row: (row["success_mean"], row["fitness_mean"]))

    lines = []
    lines.append("Stats principais por experiencia:")
    lines.append("")
    for row in summary_rows:
        lines.append(
            f"Exp {row['experiment']}: sucesso {row['success_mean']:.1%}, "
            f"fitness {row['fitness_mean']:.1f}"
        )

    lines.append("")
    lines.append(
        f"Melhor configuracao nestes testes: Experiencia {best_row['experiment']}, "
        "com maior sucesso medio e melhor fitness medio."
    )
    lines.append("")
    lines.append("O que esta a falhar mais:")
    lines.append("")
    outside_min = min(row["last_outside_pad_mean"] for row in summary_rows)
    outside_max = max(row["last_outside_pad_mean"] for row in summary_rows)
    fast_min = min(row["last_falling_too_fast_mean"] for row in summary_rows)
    fast_max = max(row["last_falling_too_fast_mean"] for row in summary_rows)
    legs_min = min(row["last_legs_not_touching_mean"] for row in summary_rows)
    legs_max = max(row["last_legs_not_touching_mean"] for row in summary_rows)
    angle_min = min(row["last_bad_angle_mean"] for row in summary_rows)
    angle_max = max(row["last_bad_angle_mean"] for row in summary_rows)
    lines.append(f"fora da plataforma: ~{outside_min:.0%} a {outside_max:.0%}")
    lines.append(f"queda demasiado rapida: ~{fast_min:.0%} a {fast_max:.0%}")
    lines.append(f"pernas sem tocar: ~{legs_min:.0%} a {legs_max:.0%}")
    lines.append(f"mau angulo: ~{angle_min:.0%} a {angle_max:.0%}")
    lines.append("")
    lines.append(
        "Conclusao pratica: o maior problema ainda e controlo horizontal / "
        "aterragem fora da plataforma, seguido de velocidade vertical demasiado alta."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ensure_plot_dir(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)


def prepare_matplotlib(plot_dir):
    ensure_plot_dir(plot_dir)
    cache_dir = os.path.abspath(os.path.join(plot_dir, ".cache"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(plot_dir, ".matplotlib")))
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def sort_experiment_key(value):
    if str(value).isdigit():
        return (0, int(value))
    return (1, str(value))


def plot_training_curves(training_rows, plot_dir):
    prepare_matplotlib(plot_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {}
    for row in training_rows:
        key = (row["experiment"], row["run"])
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(12, 7))
    for (experiment, run), rows in sorted(grouped.items(), key=lambda item: (sort_experiment_key(item[0][0]), item[0][1])):
        rows = sorted(rows, key=lambda row: row["generation"])
        generations = [row["generation"] for row in rows]
        fitness = [row["train_fitness"] for row in rows]
        plt.plot(generations, fitness, alpha=0.55, linewidth=1.2, label=f"exp {experiment} run {run}")

    plt.xlabel("Geracao")
    plt.ylabel("Fitness de treino")
    plt.title("Evolucao do melhor fitness por run")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_curves_by_run.png"), dpi=160)
    plt.close()

    experiments = sorted({row["experiment"] for row in training_rows}, key=sort_experiment_key)
    plt.figure(figsize=(12, 7))
    for experiment in experiments:
        exp_rows = [row for row in training_rows if row["experiment"] == experiment]
        generations = sorted({row["generation"] for row in exp_rows})
        means = []
        stds = []
        for generation in generations:
            values = np.array([row["train_fitness"] for row in exp_rows if row["generation"] == generation])
            means.append(values.mean())
            stds.append(values.std())
        means = np.array(means)
        stds = np.array(stds)
        plt.plot(generations, means, linewidth=2, label=f"exp {experiment}")
        plt.fill_between(generations, means - stds, means + stds, alpha=0.12)

    plt.xlabel("Geracao")
    plt.ylabel("Fitness medio de treino")
    plt.title("Comparacao media das experiencias")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_mean_by_experiment.png"), dpi=160)
    plt.close()

    final_rows = []
    for (experiment, run), rows in grouped.items():
        last = max(rows, key=lambda row: row["generation"])
        final_rows.append(last)

    experiments = sorted({row["experiment"] for row in final_rows}, key=sort_experiment_key)
    means = []
    stds = []
    for experiment in experiments:
        values = np.array([row["train_fitness"] for row in final_rows if row["experiment"] == experiment])
        means.append(values.mean())
        stds.append(values.std())

    plt.figure(figsize=(10, 6))
    x = np.arange(len(experiments))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, [f"exp {experiment}" for experiment in experiments])
    plt.ylabel("Fitness final medio")
    plt.title("Fitness final por experiencia")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_final_fitness_by_experiment.png"), dpi=160)
    plt.close()


def plot_test_summary(results, plot_dir):
    prepare_matplotlib(plot_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {}
    for result in results:
        experiment_id = experiment_id_from_filename(result["filename"])
        grouped.setdefault(experiment_id, []).append(result)

    experiments = sorted(grouped.keys(), key=sort_experiment_key)
    success_means = []
    success_stds = []
    fitness_means = []
    fitness_stds = []

    for experiment in experiments:
        success_values = np.array([r["success_rate"] for r in grouped[experiment]])
        fitness_values = np.array([r["test_avg_fitness"] for r in grouped[experiment]])
        success_means.append(success_values.mean())
        success_stds.append(success_values.std())
        fitness_means.append(fitness_values.mean())
        fitness_stds.append(fitness_values.std())

    x = np.arange(len(experiments))

    plt.figure(figsize=(10, 6))
    plt.bar(x, success_means, yerr=success_stds, capsize=5)
    plt.xticks(x, [f"exp {experiment}" for experiment in experiments])
    plt.ylabel("Taxa de sucesso")
    plt.ylim(0, 1)
    plt.title("Taxa de sucesso por experiencia")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "test_success_rate_by_experiment.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(x, fitness_means, yerr=fitness_stds, capsize=5)
    plt.xticks(x, [f"exp {experiment}" for experiment in experiments])
    plt.ylabel("Fitness medio de teste")
    plt.title("Fitness de teste por experiencia")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "test_fitness_by_experiment.png"), dpi=160)
    plt.close()


def print_experiment_stats(results):
    grouped = {}
    for result in results:
        experiment_id = experiment_id_from_filename(result["filename"])
        grouped.setdefault(experiment_id, []).append(result)

    print("\nResumo por experiencia")
    print("exp runs success_mean success_std fitness_mean fitness_std reward_mean best_log")

    def sort_key(item):
        experiment_id = item[0]
        if str(experiment_id).isdigit():
            return (0, int(experiment_id))
        return (1, str(experiment_id))

    for experiment_id, experiment_results in sorted(grouped.items(), key=sort_key):
        success_rates = np.array([r["success_rate"] for r in experiment_results])
        fitness_values = np.array([r["test_avg_fitness"] for r in experiment_results])
        reward_values = np.array([r["test_avg_reward"] for r in experiment_results])
        best = max(experiment_results, key=lambda r: r["success_rate"])

        print(
            f"{experiment_id:>3} "
            f"{len(experiment_results):>4} "
            f"{success_rates.mean():>12.1%} "
            f"{success_rates.std():>11.1%} "
            f"{fitness_values.mean():>12.3f} "
            f"{fitness_values.std():>11.3f} "
            f"{reward_values.mean():>11.3f} "
            f"{best['filename']}"
        )

    print("\nPrincipais razoes de falha no ultimo estado")
    print("exp legs_not_touching outside_pad falling_too_fast bad_angle")
    for experiment_id, experiment_results in sorted(grouped.items(), key=sort_key):
        totals = Counter()
        episodes = 0
        for result in experiment_results:
            totals.update(result["failures_last"])
            episodes += result["episodes"]
        print(
            f"{experiment_id:>3} "
            f"{totals.get('legs_not_touching', 0) / episodes:>17.1%} "
            f"{totals.get('outside_pad', 0) / episodes:>11.1%} "
            f"{totals.get('falling_too_fast', 0) / episodes:>16.1%} "
            f"{totals.get('bad_angle', 0) / episodes:>9.1%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Specific log file to debug. By default, uses all log*.txt files.")
    parser.add_argument("--mode", choices=["best", "last"], default="best")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--show-cases", type=int, default=3)
    parser.add_argument("--summary", action="store_true", help="Print aggregate stats grouped by experiment.")
    parser.add_argument("--train-only", action="store_true", help="Only analyze fitness values stored in the training logs.")
    parser.add_argument("--plots", action="store_true", help="Create matplotlib plots in the plot directory.")
    parser.add_argument("--plot-dir", default="plots", help="Directory where plots and CSV files are written.")
    parser.add_argument("--log-root", default="logs", help="Directory where experiment log folders are stored.")
    parser.add_argument("--csv", action="store_true", help="Write CSV files with the computed statistics.")
    args = parser.parse_args()

    lander = load_project()

    if args.log is not None:
        logs = [args.log]
    else:
        logs = find_log_files(args.log_root)

    if not logs:
        raise SystemExit("No log files found.")

    if args.train_only:
        training_rows = []
        for filename in logs:
            training_rows.extend(load_training_curve(lander, filename))

        final_rows = []
        for filename in logs:
            rows = [row for row in training_rows if row["filename"] == filename]
            if rows:
                final_rows.append(max(rows, key=lambda row: row["generation"]))

        print("\nResumo rapido dos logs de treino")
        print("log experiment run generations final_train_fitness best_train_fitness")
        for filename in logs:
            rows = [row for row in training_rows if row["filename"] == filename]
            if not rows:
                continue
            final = max(rows, key=lambda row: row["generation"])
            best = max(rows, key=lambda row: row["train_fitness"])
            print(
                f"{filename} "
                f"{final['experiment']} "
                f"{final['run']} "
                f"{final['generation'] + 1} "
                f"{final['train_fitness']:.3f} "
                f"{best['train_fitness']:.3f}"
            )

        if args.csv:
            ensure_plot_dir(args.plot_dir)
            write_csv(
                os.path.join(args.plot_dir, "training_curves.csv"),
                training_rows,
                ["filename", "experiment", "run", "generation", "train_fitness"],
            )
            write_csv(
                os.path.join(args.plot_dir, "training_final.csv"),
                final_rows,
                ["filename", "experiment", "run", "generation", "train_fitness"],
            )

        if args.plots:
            plot_training_curves(training_rows, args.plot_dir)
            print(f"\nGraficos guardados em: {args.plot_dir}")

        return

    results = []
    for filename in logs:
        results.append(summarize_controller(lander, filename, args.mode, args.episodes, args.show_cases))

    if args.summary:
        print_experiment_stats(results)

    if args.csv:
        ensure_plot_dir(args.plot_dir)
        summary_rows = summarize_results_by_experiment(results)
        write_csv(
            os.path.join(args.plot_dir, "test_summary.csv"),
            [flatten_test_result(result) for result in results],
            [
                "filename",
                "train_fitness",
                "test_avg_fitness",
                "test_avg_reward",
                "success_rate",
                "successes",
                "episodes",
                "last_success",
                "last_legs_not_touching",
                "last_outside_pad",
                "last_falling_too_fast",
                "last_bad_angle",
            ],
        )
        write_csv(
            os.path.join(args.plot_dir, "experiment_summary.csv"),
            summary_rows,
            [
                "experiment",
                "runs",
                "success_mean",
                "success_std",
                "fitness_mean",
                "fitness_std",
                "reward_mean",
                "last_legs_not_touching_mean",
                "last_outside_pad_mean",
                "last_falling_too_fast_mean",
                "last_bad_angle_mean",
                "best_log",
            ],
        )
        write_experiment_summary_txt(
            os.path.join(args.plot_dir, "experiment_summary.txt"),
            summary_rows,
        )

    if args.plots:
        plot_training_curves(
            [row for filename in logs for row in load_training_curve(lander, filename)],
            args.plot_dir,
        )
        if args.summary:
            plot_test_summary(results, args.plot_dir)
        print(f"\nGraficos guardados em: {args.plot_dir}")


if __name__ == "__main__":
    main()
