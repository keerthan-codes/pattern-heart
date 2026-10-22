const base = import.meta.env.VITE_API_BASE_URL || '/api'
export async function api(path, options = {}) {
  let response
  try {
    response = await fetch(`${base}${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers }, signal: AbortSignal.timeout(20000) })
  } catch {
    throw new Error('Cannot reach the backend. Start Flask on port 5001 and try again.')
  }
  let body
  try { body = await response.json() } catch { throw new Error('The backend returned an unreadable response. Check that Flask is running on port 5001.') }
  if (!response.ok) {
    const error = new Error(body.error || 'The request could not be completed.')
    error.fields = body.fields
    throw error
  }
  return body
}
