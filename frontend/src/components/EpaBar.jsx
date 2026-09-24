import DecisionIcon from './DecisionIcon'

const DECISION_META = {
  GO: { label: 'Go for it', colorVar: '--go' },
  FIELD_GOAL: { label: 'Field goal', colorVar: '--field-goal' },
  PUNT: { label: 'Punt', colorVar: '--punt' },
}

/**
 * One row of a 3-way value comparison (used for both the EPA and the win-
 * probability panels). `pct` (0-100, computed once by the caller) is how
 * close this option is to the best one in its panel, not its raw distance
 * from zero -- so the recommended row always renders as the fullest bar,
 * and the others show proportionally how far behind it they are. The exact
 * signed value is still shown as text regardless.
 */
export default function EpaBar({
  decision,
  value,
  outOfRange,
  isWinner,
  pct,
  formatValue,
  metricName,
}) {
  const meta = DECISION_META[decision]
  const isFlagged = outOfRange.length > 0

  const title = `${meta.label}: ${formatValue(value)} predicted ${metricName}${
    isFlagged ? `. Outside the historical range for ${outOfRange.join(', ')}.` : ''
  }`

  return (
    <div className={`epa-row${isWinner ? ' epa-row-winner' : ''}`} title={title}>
      <div className="epa-row-head">
        <DecisionIcon
          decision={decision}
          size={16}
          className="epa-icon"
          style={{ color: `var(${meta.colorVar})` }}
        />
        <span className="epa-label">{meta.label}</span>
        {isWinner && <span className="epa-winner-tag">Recommended</span>}
        {isFlagged && (
          <span className="epa-warning-tag">
            <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
              <path
                d="M8 1.5 15 14H1L8 1.5Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinejoin="round"
              />
              <path d="M8 6v4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
              <circle cx="8" cy="11.6" r="0.9" fill="currentColor" />
            </svg>
            Outside historical range
          </span>
        )}
        <span className="epa-value">{formatValue(value)}</span>
      </div>

      <div className="epa-track">
        <span
          className={`epa-fill${isFlagged ? ' epa-fill-muted' : ''}`}
          style={{
            width: `${pct}%`,
            background: `var(${meta.colorVar})`,
          }}
        />
      </div>
    </div>
  )
}
