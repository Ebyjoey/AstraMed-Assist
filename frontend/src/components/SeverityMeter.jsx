import React from 'react';
import { motion } from 'framer-motion';

export default function SeverityMeter({ value = 0, label }) {
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
