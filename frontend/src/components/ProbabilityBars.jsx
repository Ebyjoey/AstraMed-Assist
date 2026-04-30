import React from 'react';
import { motion } from 'framer-motion';

const DISEASE_CONFIG = {
  Pneumonia:    { key: 'pneumonia', color: '#F59E0B' },
  Tuberculosis: { key: 'tb',        color: '#7C3AED' },
  Normal:       { key: 'normal',    color: '#10B981' },
};

export default function ProbabilityBars({ probabilities = {}, severities = {} }) {
  return (
    <div className="prob-bar-row">
      {Object.entries(DISEASE_CONFIG).map(([name, cfg]) => {
        const prob = probabilities[name] ?? 0;
        const sev  = severities[name]  ?? 0;
        const pct  = Math.round(prob * 100);
        const detected = prob >= 0.5;

        return (
          <div key={name} className="prob-bar-item">
            <div className="prob-bar-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: cfg.color,
                  boxShadow: detected ? `0 0 6px ${cfg.color}` : 'none',
                  transition: 'box-shadow 0.3s'
                }} />
                <span className="prob-bar-label">{name}</span>
                {detected && (
                  <span style={{
                    fontSize: 9, fontWeight: 700,
                    padding: '1px 6px', borderRadius: 10,
                    background: cfg.color + '22',
                    color: cfg.color,
                    letterSpacing: 0.3,
                  }}>
                    DETECTED
                  </span>
                )}
              </div>
              <span className="prob-bar-value" style={{ color: cfg.color }}>
                {pct}%
              </span>
            </div>

            {/* Probability bar */}
            <div className="prob-bar-track">
              <motion.div
                className="prob-bar-fill"
                style={{ background: cfg.color, opacity: 0.85 }}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.9, ease: [0.4, 0, 0.2, 1], delay: 0.05 }}
              />
            </div>

            {/* Severity sub-bar */}
            {sev > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
                <span style={{ fontSize: 9, color: 'var(--gray-400)', flex: '0 0 54px' }}>
                  Severity:
                </span>
                <div style={{ flex: 1, height: 4, background: 'var(--gray-100)', borderRadius: 2, overflow: 'hidden' }}>
                  <motion.div
                    style={{ height: '100%', background: cfg.color + '99', borderRadius: 2 }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.round(sev * 100)}%` }}
                    transition={{ duration: 0.9, ease: [0.4, 0, 0.2, 1], delay: 0.2 }}
                  />
                </div>
                <span style={{ fontSize: 9, color: 'var(--gray-500)', fontFamily: 'Space Mono, monospace', flex: '0 0 32px', textAlign: 'right' }}>
                  {Math.round(sev * 100)}%
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
