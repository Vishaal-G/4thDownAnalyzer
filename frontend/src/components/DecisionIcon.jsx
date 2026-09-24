// Small play-call glyphs, one per decision. Colored via `currentColor` so the
// parent sets `color` (usually one of the --go/--field-goal/--punt tokens) —
// reinforces identity alongside the text label and series color, never
// standing in for color alone.

function FootballIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <ellipse
        cx="12"
        cy="12"
        rx="9.5"
        ry="5.5"
        transform="rotate(-32 12 12)"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path d="M7.3 13.7 16.7 10.3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <path
        d="M9.9 12.6 8.9 14.3M11.9 11.9 10.9 13.6M13.9 11.1 12.9 12.8"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinecap="round"
      />
    </svg>
  )
}

function GoalPostIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M6 3v6M18 3v6M6 9h12M12 9v12"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function PuntIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M3 19c4-14 14-14 18 0"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeDasharray="0.5 3.2"
      />
      <circle cx="3" cy="19" r="1.6" fill="currentColor" />
    </svg>
  )
}

const ICONS = {
  GO: FootballIcon,
  FIELD_GOAL: GoalPostIcon,
  PUNT: PuntIcon,
}

export default function DecisionIcon({ decision, size = 16, ...rest }) {
  const Icon = ICONS[decision]
  return <Icon width={size} height={size} aria-hidden="true" {...rest} />
}
