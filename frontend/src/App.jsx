import { useState } from 'react'
import DecisionForm from './components/DecisionForm'
import ResultModal from './components/ResultModal'
import { getRecommendation } from './lib/api'
import './App.css'

export default function App() {
  const [status, setStatus] = useState('idle') // idle | loading | error | done
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(payload) {
    setStatus('loading')
    setError(null)
    try {
      const data = await getRecommendation(payload)
      setResult(data)
      setStatus('done')
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  function handleClose() {
    setStatus('idle')
    setResult(null)
  }

  return (
    <>
      <header className="hero">
        <p className="hero-eyebrow">4th &amp; Long</p>
        <h1>Should you go for it?</h1>
        <p className="hero-sub">
          Enter the game situation below. Three models, trained on real NFL
          play-by-play data, each estimate the expected points added if you go
          for it, kick, or punt.
        </p>
      </header>

      <main>
        <DecisionForm onSubmit={handleSubmit} isLoading={status === 'loading'} />

        {status === 'error' && (
          <div className="panel error-banner" role="alert">
            <strong>Couldn&rsquo;t get a recommendation.</strong> {error}
            <div className="error-hint">
              Is the backend running? (<code>python app.py</code> in{' '}
              <code>backend/</code>)
            </div>
          </div>
        )}
      </main>

      {status === 'done' && result && <ResultModal result={result} onClose={handleClose} />}

      <footer className="footer">
        Predictions are model estimates from historical outcomes, not guarantees.
      </footer>
    </>
  )
}
