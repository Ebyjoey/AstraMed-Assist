import React from 'react';
import { motion } from 'framer-motion';
import { AlertOctagon, AlertTriangle, CheckCircle } from 'lucide-react';

// ── Triage Badge ─────────────────────────────────────────────────────────────

const TRIAGE_CONFIG = {
  High:   { cls: 'triage-high',   Icon: AlertOctagon,  pulse: true  },
  Medium: { cls: 'triage-medium', Icon: AlertTriangle,  pulse: false },
  Low:    { cls: 'triage-low',    Icon: CheckCircle,    pulse: false },
};

export default function TriageBadge({ level = 'Low', score }) {
  const cfg = TRIAGE_CONFIG[level] || TRIAGE_CONFIG.Low;
  const { cls, Icon, pulse } = cfg;

  return (
    <motion.div
      className={`triage-badge ${cls}`}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 20 }}
    >
      {pulse ? (
        <span style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
          <Icon size={16} />
          <motion.span
            style={{
              position: 'absolute', inset: -4,
              borderRadius: '50%',
              border: '2px solid currentColor',
              opacity: 0.4,
            }}
            animate={{ scale: [1, 1.6], opacity: [0.4, 0] }}
            transition={{ duration: 1.2, repeat: Infinity }}
          />
        </span>
      ) : (
        <Icon size={16} />
      )}
      <span>{level} Priority</span>
      {score !== undefined && (
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 12, opacity: 0.7 }}>
          {score.toFixed(3)}
        </span>
      )}
    </motion.div>
  );
}

// ── Severity Meter ────────────────────────────────────────────────────────────

export function SeverityMeter({ value = 0, label }) {
  const pct = Math.round(value * 100);
  return (
    <div className="severity-meter">
      <div className="severity-track">
        <motion.div
          className="severity-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: [0.4, 0, 0.2, 1] }}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
        <div className="severity-labels" style={{ flex: 1 }}>
          <span>Minimal</span>
          <span>Mild</span>
          <span>Moderate</span>
          <span>Severe</span>
        </div>
        <span style={{
          fontFamily: 'Space Mono, monospace', fontSize: 12, fontWeight: 700,
          color: pct > 60 ? 'var(--red-600)' : pct > 35 ? 'var(--amber-600)' : 'var(--green-600)',
          marginLeft: 10
        }}>
          {label || (pct > 60 ? 'Severe' : pct > 35 ? 'Moderate' : pct > 15 ? 'Mild' : 'Minimal')}
        </span>
      </div>
    </div>
  );
}
