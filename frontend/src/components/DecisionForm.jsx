import { useMemo, useState } from 'react'

function yardlineHelper(yardline100) {
  const n = Number(yardline100)
  if (!Number.isFinite(n) || n < 1 || n > 99) return null
  if (n === 50) return "→ midfield"
  return n < 50 ? `→ opponent's ${n}` : `→ your own ${100 - n}`
}

function spotLabel(yardline100) {
  const n = Number(yardline100)
  if (!Number.isFinite(n) || n < 1 || n > 99) return null
  if (n === 50) return 'midfield'
  return n < 50 ? `opp ${n}` : `own ${100 - n}`
}

export default function DecisionForm({ onSubmit, isLoading }) {
  const [yardline100, setYardline100] = useState('60')
  const [ydstogo, setYdstogo] = useState('4')
  const [scoreMode, setScoreMode] = useState('ahead') // 'ahead' | 'behind'
  const [scoreMargin, setScoreMargin] = useState('0')
  const [minutes, setMinutes] = useState('12')
  const [seconds, setSeconds] = useState('0')

  const helper = useMemo(() => yardlineHelper(yardline100), [yardline100])
  const spot = useMemo(() => spotLabel(yardline100), [yardline100])

  function handleSubmit(e) {
    e.preventDefault()

    const totalSeconds = Math.max(
      0,
      Math.min(3600, (Number(minutes) || 0) * 60 + (Number(seconds) || 0)),
    )

    const signedScoreDiff = (scoreMode === 'behind' ? -1 : 1) * Math.abs(Number(scoreMargin) || 0)

    onSubmit({
      yardline_100: Number(yardline100),
      ydstogo: Number(ydstogo),
      score_differential: signedScoreDiff,
      game_seconds_remaining: totalSeconds,
    })
  }

  return (
    <form className="panel form" onSubmit={handleSubmit}>
      <div className="down-badge">
        <span className="down-badge-down">4th</span>
        <span className="down-badge-and">&amp;</span>
        <span className="down-badge-yards">{ydstogo || '0'}</span>
        {spot && <span className="down-badge-spot">{spot}</span>}
      </div>

      <div className="field">
        <label htmlFor="yardline">Yard line</label>
        <input
          id="yardline"
          type="number"
          min="1"
          max="99"
          required
          value={yardline100}
          onChange={(e) => setYardline100(e.target.value)}
        />
        <span className="field-help">
          {helper ?? 'Distance from the opponent’s end zone, 1 to 99'}
        </span>
      </div>

      <div className="field">
        <label htmlFor="ydstogo">Yards to go</label>
        <input
          id="ydstogo"
          type="number"
          min="1"
          max="99"
          required
          value={ydstogo}
          onChange={(e) => setYdstogo(e.target.value)}
        />
        <span className="field-help">For the first down</span>
      </div>

      <div className="field">
        <span className="field-label-static">Score</span>
        <div className="score-input-row">
          <div className="score-toggle" role="group" aria-label="Ahead or behind">
            <button
              type="button"
              className={`score-toggle-btn${scoreMode === 'ahead' ? ' active' : ''}`}
              onClick={() => setScoreMode('ahead')}
            >
              Ahead
            </button>
            <button
              type="button"
              className={`score-toggle-btn${scoreMode === 'behind' ? ' active' : ''}`}
              onClick={() => setScoreMode('behind')}
            >
              Behind
            </button>
          </div>
          <input
            type="number"
            min="0"
            required
            aria-label="By how many points"
            value={scoreMargin}
            onChange={(e) => setScoreMargin(e.target.value)}
          />
        </div>
        <span className="field-help">By how many points</span>
      </div>

      <div className="field">
        <span className="field-label-static">Time remaining</span>
        <div className="time-inputs">
          <input
            type="number"
            min="0"
            max="60"
            aria-label="Minutes remaining"
            value={minutes}
            onChange={(e) => setMinutes(e.target.value)}
          />
          <span className="time-sep">:</span>
          <input
            type="number"
            min="0"
            max="59"
            aria-label="Seconds remaining"
            value={seconds}
            onChange={(e) => setSeconds(e.target.value)}
          />
        </div>
        <span className="field-help">In the game, not the quarter</span>
      </div>

      <button type="submit" className="submit-btn" disabled={isLoading}>
        {isLoading ? 'Making the call…' : 'Make the call'}
      </button>
    </form>
  )
}
