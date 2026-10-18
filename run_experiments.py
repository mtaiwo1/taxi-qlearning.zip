"""Runs every experiment for the assignment and saves plots + metrics to results/."""
import json, warnings
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlearning import train, evaluate

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 5000
BASE = dict(alpha=0.1, gamma=0.99, eps_start=1.0, eps_min=0.05, eps_decay=0.995)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
SELECT_SEED, FINAL_SEED = 5000, 9000  # eval seeds: selection vs final report (kept separate)


def job(args):
    name, cfg, seed, noise_p = args
    agent, rewards = train(**cfg, episodes=EPISODES, seed=seed, noise_p=noise_p)
    return name, seed, rewards, agent.Q


def smooth(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def run_group(pool, configs, noise_p=0.0):
    """configs: {name: cfg}. Returns {name: (rewards[seeds, eps], [Q per seed])}."""
    jobs = [(n, c, s, noise_p) for n, c in configs.items() for s in SEEDS]
    out = {n: ([None] * len(SEEDS), [None] * len(SEEDS)) for n in configs}
    for n, s, r, Q in pool.map(job, jobs):
        out[n][0][SEEDS.index(s)] = r
        out[n][1][SEEDS.index(s)] = Q
    return {n: (np.array(v[0]), v[1]) for n, v in out.items()}


def curve_plot(results, title, fname, labels=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (n, (R, _)) in enumerate(results.items()):
        sm = np.array([smooth(r) for r in R])
        m, sd = sm.mean(0), sm.std(0)
        x = np.arange(len(m)) + 100
        ax.plot(x, m, color=COLORS[i % 5], label=(labels or {}).get(n, n), lw=1.8)
        ax.fill_between(x, m - sd, m + sd, color=COLORS[i % 5], alpha=0.15, lw=0)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Reward per episode (100-ep moving avg)")
    ax.set_title(title); ax.set_ylim(-250, 20); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"results/{fname}", dpi=160); plt.close(fig)


def summarize(results, noise_p=0.0, seed0=SELECT_SEED):
    rows = {}
    for n, (R, Qs) in results.items():
        ev = [evaluate(Q, 100, seed0, noise_p) for Q in Qs]
        rows[n] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
        rows[n]["seed_std_of_mean_reward"] = float(np.std([e["mean_reward"] for e in ev]))
        rows[n]["train_last500_mean"] = float(R[:, -500:].mean())
        # episodes until 100-ep moving avg first exceeds 0 (avg over seeds that get there)
        hit = [int(np.argmax(smooth(r) > 0)) + 100 if (smooth(r) > 0).any() else np.nan for r in R]
        rows[n]["episodes_to_positive_avg"] = float(np.nanmean(hit)) if not np.all(np.isnan(hit)) else None
    return rows


if __name__ == "__main__":
    configs = {"baseline": BASE}
    for a in [0.01, 0.5, 0.9]:
        configs[f"alpha={a}"] = {**BASE, "alpha": a}
    for g in [0.5, 0.9]:
        configs[f"gamma={g}"] = {**BASE, "gamma": g}
    for d in [0.99, 0.999]:
        configs[f"decay={d}"] = {**BASE, "eps_decay": d}

    with Pool() as pool:
        res = run_group(pool, configs)
        summ = summarize(res)
        json.dump(summ, open("results/selection_metrics.json", "w"), indent=2)

        # (C) baseline
        curve_plot({"baseline": res["baseline"]}, "Baseline: alpha=0.1, gamma=0.99, decay=0.995", "baseline.png",
                   {"baseline": "baseline (mean ± std over 5 seeds)"})
        # (D) alpha
        sub = {k: res[k] for k in ["alpha=0.01", "baseline", "alpha=0.5", "alpha=0.9"]}
        curve_plot(sub, "Learning-rate experiment", "alpha.png", {"baseline": "alpha=0.1"})
        # (E) gamma
        sub = {k: res[k] for k in ["gamma=0.5", "gamma=0.9", "baseline"]}
        curve_plot(sub, "Discount-factor experiment", "gamma.png", {"baseline": "gamma=0.99"})
        # (F) epsilon decay
        sub = {k: res[k] for k in ["decay=0.99", "baseline", "decay=0.999"]}
        curve_plot(sub, "Epsilon-decay experiment (eps_min=0.05)", "eps_decay.png", {"baseline": "decay=0.995"})

        # (G) best config by greedy eval on selection seeds
        best = max(summ, key=lambda n: (round(summ[n]["mean_reward"], 3), summ[n]["success_rate"], -summ[n]["mean_steps"]))
        print("BEST:", best, summ[best])
        final = {}
        for n in sorted({"baseline", best}):
            ev = [evaluate(Q, 100, FINAL_SEED, 0.0) for Q in res[n][1]]
            final[n] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
            final[n]["per_seed_mean_reward"] = [e["mean_reward"] for e in ev]
            final[n]["n_eval_episodes"] = 100 * len(SEEDS)
        json.dump({"best": best, "best_cfg": configs[best], "final_eval": final}, open("results/final_eval.json", "w"), indent=2)
        np.save("results/best_Q.npy", res[best][1][0])  # seed-0 agent used for video

        # (H) stochastic environment with the best config
        bcfg = configs[best]
        noise_levels = [0.0, 0.10, 0.25]
        noise_res = {}
        for p in noise_levels:
            noise_res[p] = run_group(pool, {f"p={p}": bcfg}, noise_p=p)[f"p={p}"]
        curve_plot({f"p={p}": noise_res[p] for p in noise_levels}, "Training under action noise", "noise_training.png")
        H = {"trained_and_tested_at_same_p": {}, "p0_policy_under_noise": {}}
        for p in noise_levels:
            ev = [evaluate(Q, 100, FINAL_SEED, p) for Q in noise_res[p][1]]
            H["trained_and_tested_at_same_p"][str(p)] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
            ev0 = [evaluate(Q, 100, FINAL_SEED, p) for Q in noise_res[0.0][1]]
            H["p0_policy_under_noise"][str(p)] = {k: float(np.mean([e[k] for e in ev0])) for k in ev0[0]}
            H["train_last500_mean_reward"] = H.get("train_last500_mean_reward", {})
            H["train_last500_mean_reward"][str(p)] = float(noise_res[p][0][:, -500:].mean())
        json.dump(H, open("results/noise_eval.json", "w"), indent=2)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(3); w = 0.38
        a = [H["p0_policy_under_noise"][str(p)]["mean_reward"] for p in noise_levels]
        b = [H["trained_and_tested_at_same_p"][str(p)]["mean_reward"] for p in noise_levels]
        ax.bar(x - w / 2, a, w, color=COLORS[0], label="trained at p=0")
        ax.bar(x + w / 2, b, w, color=COLORS[1], label="trained at test-time p")
        ax.set_xticks(x); ax.set_xticklabels([f"p={p}" for p in noise_levels])
        ax.set_xlabel("Action-noise level at evaluation"); ax.set_ylabel("Mean greedy reward")
        ax.set_title("Robustness to action noise"); ax.grid(axis="y", alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig("results/noise_robustness.png", dpi=160); plt.close(fig)

    print(json.dumps(summ, indent=1)); print(json.dumps(final, indent=1)); print(json.dumps(H, indent=1))
