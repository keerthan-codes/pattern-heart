import assert from 'node:assert/strict'
import { test } from 'node:test'
import { api, apiUrl } from '../src/services/api.js'

const origin = 'https://pattern-heart-api.onrender.com'

test('API URLs keep exactly one /api prefix for every endpoint and supported base', () => {
  for (const base of [undefined, '', ' ', origin, `${origin}/`, `${origin}/api`, `${origin}/api/`]) {
    for (const endpoint of ['health', 'features', 'models', 'metrics', 'predict']) {
      for (const path of [endpoint, `/${endpoint}`, `/api/${endpoint}`]) {
        assert.equal(apiUrl(path, base), `${base?.trim() ? origin : ''}/api/${endpoint}`)
      }
    }
  }
})

test('shared API client preserves JSON prediction requests and responses', async () => {
  const previousFetch = globalThis.fetch
  const prediction = { preferred: 'naive_bayes', predictions: {} }
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, '/api/predict')
      assert.equal(options.method, 'POST')
      assert.equal(options.headers['Content-Type'], 'application/json')
      assert.equal(options.body, '{"age":63}')
      return { ok: true, json: async () => prediction }
    }
    assert.deepEqual(await api('/predict', { method: 'POST', body: '{"age":63}' }), prediction)
  } finally { globalThis.fetch = previousFetch }
})
