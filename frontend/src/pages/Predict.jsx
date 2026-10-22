import { useApi } from '../services/useApi'
import { useEffect, useRef, useState } from 'react'
import { ArrowRight, Award, LoaderCircle } from 'lucide-react'
import { api } from '../services/api'
import { Status, Disclaimer } from '../components/Shared'
export default function Predict() {
  const remote = useApi('/features')
  const [values, setValues] = useState({})
  const [fields, setFields] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const resultsRef = useRef(null)
  const requestVersion = useRef(0)

  useEffect(() => () => { requestVersion.current += 1 }, [])

  useEffect(() => {
    if (!result || !resultsRef.current) return

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    resultsRef.current.scrollIntoView({
      behavior: reducedMotion ? 'instant' : 'smooth',
      block: 'start',
    })
  }, [result])

  function updateField(name, value) {
    // A response for an earlier profile must never reappear after an edit.
    requestVersion.current += 1
    setValues(current => ({ ...current, [name]: value }))
    setResult(null)
    setError('')
    setFields({})
    setBusy(false)
  }

  async function submit(event) {
    event.preventDefault()
    const version = ++requestVersion.current
    const payload = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]))
    setBusy(true); setError(''); setResult(null); setFields({})
    try {
      const prediction = await api('/predict', { method: 'POST', body: JSON.stringify(payload) })
      if (version === requestVersion.current) setResult(prediction)
    } catch (e) {
      if (version === requestVersion.current) {
        setError(e.message)
        setFields(e.fields || {})
      }
    } finally {
      if (version === requestVersion.current) setBusy(false)
    }
  }
  const groups = remote.data ? [...new Set(remote.data.features.map(f => f.group))] : []
  return <div className="page"><div className="eyebrow">EXPLORE A CLINICAL PROFILE</div><h1>One profile. Three perspectives.</h1><p className="lead">Enter recorded measurements to see how each classifier interprets the pattern.</p><Disclaimer /><Status {...remote} />
    {remote.data && <form onSubmit={submit} className="prediction-form">{groups.map((group, i) => <fieldset className="card" key={group}><legend><span className="section-number">0{i+1}</span>{group}</legend><div className="form-grid">{remote.data.features.filter(f => f.group === group).map(f => <div className="field" key={f.name}><label htmlFor={f.name}>{f.label} {f.unit && <span>({f.unit})</span>}</label>{f.type === 'category' ? <select id={f.name} required value={values[f.name] ?? ''} aria-describedby={`${f.name}-help`} aria-invalid={!!fields[f.name]} onChange={e => updateField(f.name, e.target.value)}><option value="" disabled>Select an option</option>{f.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</select> : <input id={f.name} type="number" required min={f.min} max={f.max} step={f.integer ? 1 : 'any'} placeholder={`${f.min}–${f.max}`} value={values[f.name] ?? ''} aria-describedby={`${f.name}-help`} aria-invalid={!!fields[f.name]} onChange={e => updateField(f.name, e.target.value)} />}<small id={`${f.name}-help`}>{fields[f.name] || f.description}</small></div>)}</div></fieldset>)}<Status error={error} /><div className="form-bottom"><p>All 13 fields are required. Inputs are processed for this request and are not stored by the application.</p><button className="button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={18} /> : <ArrowRight size={18} />}{busy ? 'Evaluating profile…' : 'Generate Prediction'}</button></div></form>}
    <div className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">{result ? 'Predictions are ready. Classification results are displayed below the form.' : ''}</div>
    {result && <section ref={resultsRef} className="results" aria-labelledby="results-heading"><div className="eyebrow">CLASSIFICATION RESULTS</div><h2 id="results-heading">How the models read this profile.</h2><div className="preferred-banner"><Award size={24} /><div><strong>Preferred classifier · {result.predictions[result.preferred].name}</strong><p>{result.predictions[result.preferred].label ? 'Heart disease indicated' : 'No heart disease indicated'} by this model. Selected using training cross-validation recall, then F1.</p></div></div><div className="cards">{Object.entries(result.predictions).map(([key, p]) => <article className="card" key={key}><h3>{p.name}</h3><div className={`result-label ${p.label ? 'positive' : ''}`}>{p.label ? 'Heart disease indicated' : 'No heart disease indicated'}</div><p>A classification learned from historical records, not a clinical conclusion.</p></article>)}</div><p>Models may disagree because they learn different patterns. No confidence percentage is shown; these models have not been probability-calibrated.</p><Disclaimer /></section>}
  </div>
}
