import { AlertCircle, LoaderCircle } from 'lucide-react'
export function Status({ loading, error, retry }) {
  if (loading) return <div className="status" role="status"><LoaderCircle className="spin" size={20} /> Loading from the backend…</div>
  if (error) return <div className="error" role="alert"><AlertCircle size={20} /><span>{error}</span>{retry && <button className="text-button" onClick={retry}>Try again</button>}</div>
  return null
}
export function Disclaimer() {
  return <p className="disclaimer"><AlertCircle size={17} /><span>For learning, not diagnosis. These models cannot determine whether you are healthy or replace assessment by a qualified clinician.</span></p>
}
