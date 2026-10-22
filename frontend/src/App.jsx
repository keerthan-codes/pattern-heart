import { BrowserRouter, NavLink, Link, Route, Routes, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { HeartPulse } from 'lucide-react'
import Overview from './pages/Overview'
import Predict from './pages/Predict'
import Compare from './pages/Compare'
import './App.css'
function Layout() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return <><a className="skip" href="#main">Skip to content</a><header><nav aria-label="Main navigation"><Link className="brand" to="/"><HeartPulse size={24} strokeWidth={1.8} /><span>Pattern<span className="brand-light"> / Heart</span></span></Link><div className="nav-links"><NavLink to="/" end>Overview</NavLink><NavLink to="/predict">Predict</NavLink><NavLink to="/compare">Compare Models</NavLink></div><span className="nav-note">Pattern Recognition · Academic Project</span></nav></header><main id="main"><Routes><Route path="/" element={<Overview />} /><Route path="/predict" element={<Predict />} /><Route path="/compare" element={<Compare />} /><Route path="*" element={<div className="page"><h1>Page not found.</h1><Link to="/">Return to overview</Link></div>} /></Routes></main><footer><Link className="brand" to="/"><HeartPulse size={18} />Pattern / Heart</Link><span>Comparative pattern recognition · Educational use only</span><a href="https://archive.ics.uci.edu/dataset/45/heart+disease">UCI Cleveland dataset ↗</a></footer></>
}
export default function App() { return <BrowserRouter><Layout /></BrowserRouter> }
