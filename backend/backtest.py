"""
Stage 8: backtest -- compare model recommendations against what coaches
actually did historically, on the held-out test split (season > 2021, same
split the models were validated against in Stages 4/9 -- never trained on).

For every test-set 4th down play, feed its real situation into recommend()
(EPA) and recommend_wp() (WP) and compare the model's BEST to the real
`decision` column. Reports:
  - overall agreement rate, and agreement rate broken down by what the coach
    actually did (reveals asymmetries -- e.g. the "coaches over-punt"
    hypothesis this whole project was built around, see CLAUDE.md's
    Confirmed Design Decision #1)
  - a full confusion matrix (actual decision x model's recommended decision)
  - for disagreements: the value gap (model's predicted value for its own
    pick minus its predicted value for what the coach actually did) -- this
    is the model's own estimate of "how much better" its call would have
    been, not an independent measurement
  - the biggest individual disagreements, for a look at what kind of
    situations they are

Caveat baked into every number here: each model only saw rows where that
exact decision was made historically, so evaluating it on a play where a
*different* decision was made is extrapolation for that model, same
selection-bias risk already documented in Stage 5. The out-of-range guardrail
catches only the worst of it.
"""

import pandas as pd

from decision_engine import recommend, recommend_wp

pd.set_option('display.width', 120)


def run_backtest(df, recommend_fn, value_key):
    rows = []
    for row in df.itertuples():
        result = recommend_fn(
            row.yardline_100, row.ydstogo, row.score_differential, row.game_seconds_remaining
        )
        actual = row.decision
        best = result['BEST']
        rows.append({
            'actual': actual,
            'model_best': best,
            'agree': actual == best,
            'value_gap': result[best][value_key] - result[actual][value_key],
            'actual_out_of_range': len(result[actual]['out_of_range']) > 0,
            'yardline_100': row.yardline_100,
            'ydstogo': row.ydstogo,
            'score_differential': row.score_differential,
            'game_seconds_remaining': row.game_seconds_remaining,
        })
    return pd.DataFrame(rows)


def summarize(results, label, value_label):
    print(f'\n{"=" * 70}\n{label}\n{"=" * 70}')
    print(f'Overall agreement rate: {results["agree"].mean():.1%} ({len(results)} test plays)')

    print('\nAgreement rate by what the coach actually did:')
    by_actual = results.groupby('actual')['agree'].agg(['mean', 'count'])
    by_actual.columns = ['agreement_rate', 'n']
    print(by_actual.to_string(float_format='{:.1%}'.format))

    print('\nConfusion matrix (rows = actual, columns = model recommends):')
    print(pd.crosstab(results['actual'], results['model_best']))

    disagreements = results[~results['agree']]
    print(f'\nOn the {len(disagreements)} disagreements, model-predicted {value_label} gap '
          f'(model pick minus what the coach actually did):')
    print(disagreements['value_gap'].describe().to_string(float_format='{:.4f}'.format))

    pct_actual_ood = disagreements['actual_out_of_range'].mean()
    print(f'\nOf those disagreements, {pct_actual_ood:.1%} involve an actual decision that was '
          f'itself flagged out-of-range for its own model (i.e. an unusual historical play -- '
          f'treat the gap for these with extra caution).')

    print(f'\nBiggest 10 disagreements by {value_label} gap:')
    cols = ['actual', 'model_best', 'value_gap', 'yardline_100', 'ydstogo',
            'score_differential', 'game_seconds_remaining', 'actual_out_of_range']
    print(disagreements.sort_values('value_gap', ascending=False).head(10)[cols].to_string(index=False))

    return results


def main():
    df = pd.read_parquet('../data/clean_4th_downs_wp.parquet')
    test_df = df[df['season'] > 2021]
    print(f'Test set (season > 2021): {len(test_df)} plays')
    print(test_df['decision'].value_counts().to_string())

    epa_results = run_backtest(test_df, recommend, 'epa')
    summarize(epa_results, 'EPA-based recommend()', 'EPA')

    wp_results = run_backtest(test_df, recommend_wp, 'wpa')
    summarize(wp_results, 'WP-based recommend_wp()', 'WPA')

    epa_results.to_csv('../data/backtest_epa.csv', index=False)
    wp_results.to_csv('../data/backtest_wp.csv', index=False)
    print('\nSaved data/backtest_epa.csv and data/backtest_wp.csv')


if __name__ == '__main__':
    main()
