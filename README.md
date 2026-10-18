# Q-learning on Gymnasium Taxi

Tabular Q-learning (written from scratch) on `Taxi` for the CPSC 4420 Gym assignment.
Experiments: baseline, learning rate, discount factor, epsilon decay, and final evaluation.
(`run_experiments.py` also runs an optional noisy-action experiment, which is not part of the report.)

Note: the installed Gymnasium (1.4.0) marks `Taxi-v3` as deprecated, so the code uses `Taxi-v4`.
Same task, same 500 states and 6 actions.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```
python run_experiments.py   # trains everything (5 seeds per config), writes plots + json to results/  (~3-4 min)
python render_video.py      # writes results/trained_agent.mp4 from the saved Q-table
python build_report.py      # writes report.pdf from the results
```

## Files
- `qlearning.py`: Q-learning agent, training loop, greedy evaluation, noisy-action wrapper
- `run_experiments.py`: all experiments (numbers and plots)
- `render_video.py`: renders the trained agent
- `results/`: plots (`*.png`), metrics (`*.json`), saved Q-table, `trained_agent.mp4`

## Hyperparameters
All of these are arguments of `train(...)` in `qlearning.py`: `alpha`, `gamma`, `eps_start`, `eps_min`,
`eps_decay`, `episodes`, `seed`, `noise_p`. Baseline: alpha=0.1, gamma=0.99, epsilon 1.0 decaying by 0.995 per
episode down to 0.05, 5000 episodes.
