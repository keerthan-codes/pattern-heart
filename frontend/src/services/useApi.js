import { useEffect, useState } from 'react'
import { api } from './api'
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: '', loading: true })
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let active = true
    api(path).then(data => { if (active) setState({ data, error: '', loading: false }) }).catch(error => { if (active) setState({ data: null, error: error.message, loading: false }) })
    return () => { active = false }
  }, [path, attempt])
  return { ...state, retry: () => { setState({ data: null, error: '', loading: true }); setAttempt(n => n + 1) } }
}
