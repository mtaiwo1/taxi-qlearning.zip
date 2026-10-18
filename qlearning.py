"""Tabular Q-learning for Gymnasium Taxi-v4 (written from scratch, no RL libraries)."""
import numpy as np
import gymnasium as gym


class NoisyActionWrapper(gym.ActionWrapper):
    """With probability p, replace the intended action with a different valid action."""

    def __init__(self, env, p=0.0, seed=None):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        if self.p > 0 and self.rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            return int(self.rng.choice(others))
        return action


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_min=0.05, eps_decay=0.995, seed=0):
        self.nS, self.nA = n_states, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))  # zero init

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        q = self.Q[s]
        # break ties randomly so unvisited states don't always pick action 0
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, terminated):
        # terminal states have value 0; truncation (time limit) is NOT terminal
        target = r if terminated else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


def make_env(noise_p=0.0, seed=0, render_mode=None):
    env = gym.make("Taxi-v4", render_mode=render_mode)
    if noise_p > 0:
        env = NoisyActionWrapper(env, p=noise_p, seed=seed)
    return env


def train(alpha=0.1, gamma=0.99, eps_start=1.0, eps_min=0.05, eps_decay=0.995,
          episodes=5000, seed=0, noise_p=0.0):
    env = make_env(noise_p, seed)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                           alpha, gamma, eps_start, eps_min, eps_decay, seed)
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        total, done = 0.0, False
        while not done:
            a = agent.act(s)
            s2, r, term, trunc, _ = env.step(a)
            agent.update(s, a, r, s2, term)
            total += r
            s = s2
            done = term or trunc
        agent.decay()
        rewards[ep] = total
    env.close()
    return agent, rewards


def evaluate(Q, episodes=100, seed=1000, noise_p=0.0):
    """Greedy evaluation (epsilon = 0). Returns dict of metrics."""
    env = make_env(noise_p, seed)
    rets, steps, succ = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total, n, done, term = 0.0, 0, False, False
        while not done:
            q = Q[s]
            a = int(np.argmax(q))
            s, r, term, trunc, _ = env.step(a)
            total += r
            n += 1
            done = term or trunc
        rets.append(total); steps.append(n); succ.append(float(term))
    env.close()
    return dict(mean_reward=float(np.mean(rets)), std_reward=float(np.std(rets)),
                success_rate=float(np.mean(succ)), mean_steps=float(np.mean(steps)))
