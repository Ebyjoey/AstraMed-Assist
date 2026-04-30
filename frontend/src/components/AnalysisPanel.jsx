import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, EyeOff, Thermometer, Activity, AlertTriangle } from 'lucide-react';
import ProbabilityBars from './ProbabilityBars';
import TriageBadge from './TriageBadge';
import SeverityMeter from './SeverityMeter';
import HeatmapViewer from './HeatmapViewer';

export default function AnalysisPanel({ results, imagePreview, showHeatmap, onToggleHeatmap }) {
  if (!imagePreview && !results) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">
          <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
            <rect x="8" y="8" width="48" height="48" rx="8" stroke="#D1D5DB" strokeWidth="2" strokeDasharray="6 4"/>
            <path d="M32 20v24M20 32h24" stroke="#D1D5DB" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </div>
        <p className="empty-state-text">No X-ray uploaded yet</p>
        <p className="empty-state-sub">Upload a chest X-ray and click Analyze to see AI predictions</p>
      </div>
    );
  }

  const heatmapSrc = results?.heatmap?.image_base64
    ? `data:image/png;base64,${results.heatmap.image_base64}`
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ── Image Viewer ── */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--gray-200)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="panel-title" style={{ margin: 0 }}>
            <Activity size={13} />
            {showHeatmap && heatmapSrc ? 'Grad-CAM Heatmap Overlay' : 'Chest X-Ray'}
          </span>
          {heatmapSrc && (
            <button className="heatmap-toggle" style={{ position: 'static', background: 'var(--gray-100)', color: 'var(--gray-700)', display: 'flex', alignItems: 'center', gap: 4 }} onClick={onToggleHeatmap}>
              {showHeatmap ? <EyeOff size={13} /> : <Eye size={13} />}
              {showHeatmap ? 'Show Original' : 'Show Heatmap'}
            </button>
          )}
        </div>

        <HeatmapViewer
          originalSrc={imagePreview}
          heatmapSrc={heatmapSrc}
          showHeatmap={showHeatmap && !!heatmapSrc}
        />

        {heatmapSrc && (
          <div style={{ padding: '8px 16px', background: 'var(--gray-50)', borderTop: '1px solid var(--gray-200)', fontSize: 11, color: 'var(--gray-500)' }}>
            🌡 Warm (red) regions indicate highest model activation · Primary finding: <strong>{results?.heatmap?.primary_class || '—'}</strong>
          </div>
        )}
      </div>

      {/* ── Triage + Severity ── */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="card"
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
            <div>
              <div className="panel-title" style={{ margin: 0, marginBottom: 8 }}>
                <AlertTriangle size={13} />
                Triage Assessment
              </div>
              <TriageBadge level={results.triage?.triage_level} score={results.triage?.triage_score} />
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 11, color: 'var(--gray-500)', marginBottom: 4 }}>Primary Finding</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--gray-900)' }}>
                {results.triage?.primary_finding || '—'}
              </div>
            </div>
          </div>

          {results.triage?.clinical_urgency && (
            <div className="alert alert-warning" style={{ marginBottom: 12 }}>
              {results.triage.clinical_urgency}
            </div>
          )}

          <div style={{ marginBottom: 4, fontSize: 11, color: 'var(--gray-500)', fontWeight: 600 }}>RADIOGRAPHIC SEVERITY</div>
          <SeverityMeter value={results.triage?.overall_severity || 0} label={results.triage?.severity_label} />
        </motion.div>
      )}

      {/* ── Disease Probabilities ── */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="card"
        >
          <div className="panel-title">
            <Thermometer size={13} />
            Disease Probabilities
          </div>
          <ProbabilityBars probabilities={results.probabilities} severities={results.severities} />
        </motion.div>
      )}

      {/* ── Metric Grid ── */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
          className="card"
        >
          <div className="panel-title">
            <Activity size={13} />
            Model Metrics
          </div>
          <div className="metric-grid">
            <MetricCard
              label="Triage Score"
              value={(results.triage?.triage_score || 0).toFixed(3)}
            />
            <MetricCard
              label="Confidence"
              value={`${((results.triage?.confidence || 0) * 100).toFixed(1)}%`}
            />
            <MetricCard
              label="Uncertainty"
              value={(results.uncertainty?.value || 0).toFixed(4)}
              sub={results.uncertainty?.label}
            />
            <MetricCard
              label="MC Passes"
              value={results.metadata?.mc_passes || 20}
              sub="stochastic"
            />
          </div>

          {/* Uncertainty bar */}
          <div style={{ marginTop: 12 }}>
            <div className="uncertainty-row">
              <span className="unc-label">Uncertainty</span>
              <div className="unc-bar-track">
                <div
                  className="unc-bar-fill"
                  style={{ width: `${Math.min((results.uncertainty?.value || 0) * 500, 100)}%` }}
                />
              </div>
              <span className="unc-value">{(results.uncertainty?.value || 0).toFixed(4)}</span>
            </div>
            {results.uncertainty?.recommendation && (
              <p style={{ fontSize: 11, color: 'var(--gray-500)', marginTop: 6 }}>
                {results.uncertainty.recommendation}
              </p>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div className="metric-card">
      <div className="metric-val">{value}</div>
      <div className="metric-label">{label}</div>
      {sub && <div style={{ fontSize: 9, color: 'var(--gray-400)', marginTop: 1 }}>{sub}</div>}
    </div>
  );
}
