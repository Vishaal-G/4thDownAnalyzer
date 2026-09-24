/**
 * Calls the Flask backend's POST /api/recommend (backend/app.py).
 *
 * Locally, VITE_API_URL is unset, so this hits a relative `/api/recommend`,
 * forwarded by Vite's dev-server proxy (see vite.config.js) as a same-origin
 * request -- no CORS involved. In production (frontend and backend on
 * separate domains, e.g. Vercel + Render), set VITE_API_URL to the deployed
 * backend's full URL as a build-time env var, and this becomes a real
 * cross-origin request instead (handled by backend/app.py's CORS(app)).
 *
 * @param {{yardline_100: number, ydstogo: number, score_differential: number, game_seconds_remaining: number}} payload
 * @returns {Promise<{GO: object, PUNT: object, FIELD_GOAL: object, BEST: string}>}
 */
const API_BASE = import.meta.env.VITE_API_URL ?? ''

export async function getRecommendation(payload) {
  const response = await fetch(`${API_BASE}/api/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  let body = null
  try {
    body = await response.json()
  } catch {
    // Non-JSON response (e.g. the dev server itself errored) — body stays null.
  }

  if (!response.ok) {
    const message = body?.error || `Request failed with status ${response.status}`
    throw new Error(message)
  }

  return body
}
