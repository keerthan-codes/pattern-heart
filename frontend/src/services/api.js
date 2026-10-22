export function apiUrl(path, baseUrl = import.meta.env?.VITE_API_BASE_URL) {
  // Accept a backend origin or an existing /api base, with optional trailing slashes.
  const origin = (baseUrl || '').trim().replace(/\/+$/, '').replace(/\/api$/, '')
  const endpoint = path.replace(/^\/+/, '').replace(/^api(?:\/|$)/, '')
  return `${origin}/api/${endpoint}`
}

export async function api(path, options = {}) {
  let response
  try {
    response = await fetch(apiUrl(path), { ...options, headers: { 'Content-Type': 'application/json', ...options.headers }, signal: AbortSignal.timeout(20000) })
  } catch {
    throw new Error('Cannot reach the prediction service. Please try again shortly.')
  }
  let body
  try { body = await response.json() } catch { throw new Error('The prediction service returned an unreadable response. Please try again shortly.') }
  if (!response.ok) {
    const error = new Error(body.error || 'The request could not be completed.')
    error.fields = body.fields
    throw error
  }
  return body
}
