import EpaBar from './EpaBar'
import DecisionIcon from './DecisionIcon'

const DECISION_ORDER = ['GO', 'FIELD_GOAL', 'PUNT']

const BEST_META = {
  GO: { label: 'Go for it', colorVar: '--go' },
  FIELD_GOAL: { label: 'Kick the field goal', colorVar: '--field-goal' },
  PUNT: { label: 'Punt', colorVar: '--punt' },
}

function formatEpa(value) {
  return `${value > 0 ? '+' : ''}${value.toFixed(3)}`
}

function formatWp(value) {
  return `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%`
}

// Bar length = how close an option is to the best one, not its raw distance
// from zero -- so the recommended option always renders as the fullest bar
// (100%), and the others show proportionally how far behind it they are.
// The exact signed value is still shown as text either way.
function relativeBarPcts(valuesByDecision, bestKey) {
  const bestValue = valuesByDecision[bestKey]
  const gaps = {}
  for (const [decision, value] of Object.entries(valuesByDecision)) {
    gaps[decision] = Math.max(0, bestValue - value)
  }
  const maxGap = Math.max(...Object.values(gaps), 0.0001)
  const pcts = {}
  for (const [decision, gap] of Object.entries(gaps)) {
    pcts[decision] = Math.max(4, 100 - (gap / maxGap) * 100)
  }
  return pcts
}

export default function ResultModal({ result, onClose }) {
  const wp = result.WP
  const epaBest = result.BEST
  const wpBest = wp.BEST
  const agree = epaBest === wpBest

  const epaValues = Object.fromEntries(DECISION_ORDER.map((d) => [d, result[d].epa]))
  const wpValues = Object.fromEntries(DECISION_ORDER.map((d) => [d, wp[d].wpa]))
  const epaPcts = relativeBarPcts(epaValues, epaBest)
  const wpPcts = relativeBarPcts(wpValues, wpBest)

  const primary = BEST_META[wpBest]
  const secondary = BEST_META[epaBest]

  return (
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="result-heading"
        onClick={(e) => e.stopPropagation()}
      >
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
          ✕
        </button>

        <p className="modal-eyebrow">The call</p>
        <h2 id="result-heading" className="modal-best" style={{ color: `var(${primary.colorVar})` }}>
          <DecisionIcon decision={wpBest} size={36} className="modal-best-icon" />
          {primary.label}
        </h2>

        {agree ? (
          <p className="modal-agree">Both models agree on this one.</p>
        ) : (
          <p className="modal-disagree" style={{ color: `var(${secondary.colorVar})` }}>
            <DecisionIcon decision={epaBest} size={13} />
            Points alone would say {secondary.label.toLowerCase()} instead
          </p>
        )}

        <div className="metric-panel">
          <p className="metric-panel-title">Win probability added</p>
          <div className="epa-list">
            {DECISION_ORDER.map((decision) => (
              <EpaBar
                key={decision}
                decision={decision}
                value={wp[decision].wpa}
                outOfRange={wp[decision].out_of_range}
                isWinner={decision === wpBest}
                pct={wpPcts[decision]}
                formatValue={formatWp}
                metricName="win probability added"
              />
            ))}
          </div>
        </div>

        <div className="metric-panel metric-panel-secondary">
          <p className="metric-panel-title">Expected points added</p>
          <div className="epa-list">
            {DECISION_ORDER.map((decision) => (
              <EpaBar
                key={decision}
                decision={decision}
                value={result[decision].epa}
                outOfRange={result[decision].out_of_range}
                isWinner={decision === epaBest}
                pct={epaPcts[decision]}
                formatValue={formatEpa}
                metricName="EPA"
              />
            ))}
          </div>
        </div>

        <p className="modal-note">
          Win probability added accounts for the score and clock, so it
          catches cases points alone misses, like a field goal that
          isn&rsquo;t enough to actually win. A flagged option wasn&rsquo;t
          considered for either recommendation, since this situation is
          outside what that model saw during training.
        </p>

        <button type="button" className="submit-btn modal-again" onClick={onClose}>
          Try another scenario
        </button>
      </div>
    </div>
  )
}
