/**
 * Calls the Flask backend's POST /api/recommend (backend/app.py), forwarded
 * by Vite's dev-server proxy (see vite.config.js) so this can just be a
 * same-origin fetch.
 *
 * @param {{yardline_100: number, ydstogo: number, score_differential: number, game_seconds_remaining: number}} payload
 * @returns {Promise<{GO: object, PUNT: object, FIELD_GOAL: object, BEST: string}>}
 */
export async function getRecommendation(payload) {
  const response = await fetch('/api/recommend', {
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
