"""
Trains 3 small PyTorch MLPs (GO / FIELD_GOAL / PUNT) predicting `wpa` (win
probability added), same methodology as notebooks/04_train.ipynb (Stage 4).

v2 (2026-09-23): added an engineered 5th feature, `score_after_fg` =
`score_differential + 3` -- the margin you'd have immediately after an
otherwise-successful field goal (>=0 means a FG alone ties/wins; negative
means you'd still be trailing by that much even after one). Added after a
concrete finding: a 4-feature WP model correctly flipped a down-6 case (FG
doesn't help) to GO, but missed a very similar down-5 case, where GO and
FIELD_GOAL were within 0.1 percentage points of each other -- essentially
noise, not a confident call. Hypothesis: the model had to infer "does a FG
even help here" indirectly from raw score_differential and had too little
dense signal to do that precisely in this narrow endgame pocket; a direct
feature for it should help. See CLAUDE.md's Stage 9 follow-up note for the
before/after comparison.

Because the feature set changed (4 -> 5 inputs), this version can no longer
reuse backend/models/scaling_stats.json (built for the EPA/4-feature models)
-- it computes and saves its own backend/models/scaling_stats_wp.json.

Input data: data/clean_4th_downs_wp.parquet (see v1 docstring history / git
blame -- unchanged from v1: clean_4th_downs.parquet with `wpa` attached from
data/raw_4th_downs.parquet by row index).

Saves weights to backend/models/model_GO_wp.pt, model_FieldGoal_wp.pt,
model_Punt_wp.pt (overwrites the v1/4-feature weights -- decision_engine.py's
recommend_wp() is updated alongside this to match the new 5-input shape).
"""

import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
MODELS_DIR = Path(__file__).parent / 'models'
BASE_FEATURES = ['yardline_100', 'ydstogo', 'score_differential', 'game_seconds_remaining']
FEATURE_NAMES = BASE_FEATURES + ['score_after_fg']

DECISION_FILES = {
    'GO': 'model_GO_wp.pt',
    'PUNT': 'model_Punt_wp.pt',
    'FIELD_GOAL': 'model_FieldGoal_wp.pt',
}


def add_engineered_features(df):
    df = df.copy()
    df['score_after_fg'] = df['score_differential'] + 3
    return df


def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_NAMES), 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )


def compute_scaling_stats(train_df):
    mean = train_df[FEATURE_NAMES].mean()
    std = train_df[FEATURE_NAMES].std()
    minmax = train_df[FEATURE_NAMES].agg(['min', 'max'])
    return {
        'mean': mean.to_dict(),
        'std': std.to_dict(),
        'min': minmax.loc['min'].to_dict(),
        'max': minmax.loc['max'].to_dict(),
    }


def to_tensor(df, mean, std):
    scaled = pd.DataFrame({f: (df[f] - mean[f]) / std[f] for f in FEATURE_NAMES})
    x = torch.tensor(scaled[FEATURE_NAMES].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df['wpa'].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    return x, y


def train_one(df, label, epochs=4000, lr=0.001):
    sub = df[df['decision'] == label]
    train_df = sub[sub['season'] <= 2021]
    test_df = sub[sub['season'] > 2021]

    stats = compute_scaling_stats(train_df)
    mean, std = stats['mean'], stats['std']

    x_train, y_train = to_tensor(train_df, mean, std)
    x_test, y_test = to_tensor(test_df, mean, std)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    naive_baseline_mse = torch.mean((y_test - y_train.mean()) ** 2).item()

    train_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        train_loss = torch.mean((pred - y_train) ** 2)
        train_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_mse = torch.mean((model(x_test) - y_test) ** 2).item()

    print(
        f'{label}: train rows={len(train_df)} test rows={len(test_df)} '
        f'final train MSE={train_loss.item():.6f} test MSE={test_mse:.6f} '
        f'naive baseline test MSE={naive_baseline_mse:.6f} '
        f'(beats baseline: {test_mse < naive_baseline_mse})'
    )

    torch.save(model.state_dict(), MODELS_DIR / DECISION_FILES[label])
    return model, stats


def main():
    df = pd.read_parquet(ROOT / 'data' / 'clean_4th_downs_wp.parquet')
    df = add_engineered_features(df)

    all_stats = {}
    for label in ['GO', 'FIELD_GOAL', 'PUNT']:
        _model, stats = train_one(df, label)
        all_stats[label] = stats

    with open(MODELS_DIR / 'scaling_stats_wp.json', 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    print('saved backend/models/scaling_stats_wp.json')


if __name__ == '__main__':
    main()
