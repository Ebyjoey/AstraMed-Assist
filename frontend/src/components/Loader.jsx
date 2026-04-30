import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const STEPS = [
  'Preprocessing X-ray image...',
  'Extracting DenseNet-121 features...',
  'Running multi-label classification...',
  'Computing Grad-CAM heatmaps...',
  'Estimating severity scores...',
  'Running Monte Carlo Dropout...',
  'Computing triage score...',
  'Finalizing results...',
];

export default function Loader({ message }) {
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStepIdx(i => (i + 1) % STEPS.length);
    }, 900);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loader-wrap">
      {/* Animated rings */}
      <div style={{ position: 'relative', width: 72, height: 72 }}>
        <motion.div
          style={{
            position: 'absolute', inset: 0,
            border: '3px solid var(--brand-blue-pale)',
            borderRadius: '50%',
          }}
        />
        <motion.div
          style={{
            position: 'absolute', inset: 0,
            border: '3px solid transparent',
            borderTopColor: 'var(--brand-blue)',
            borderRadius: '50%',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          style={{
            position: 'absolute', inset: 10,
            border: '2px solid transparent',
            borderTopColor: 'var(--brand-blue)',
            borderRadius: '50%',
            opacity: 0.5,
          }}
          animate={{ rotate: -360 }}
          transition={{ duration: 1.3, repeat: Infinity, ease: 'linear' }}
        />
        {/* Center icon */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2v20M2 12h20" stroke="var(--brand-blue)" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="12" cy="12" r="5" stroke="var(--brand-blue)" strokeWidth="1.5" fill="none"/>
          </svg>
        </div>
      </div>

      {/* Main message */}
      <p className="loader-msg">{message || 'Analyzing X-ray...'}</p>

      {/* Rotating step label */}
      <AnimatePresence mode="wait">
        <motion.p
          key={stepIdx}
          className="loader-sub"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
        >
          {STEPS[stepIdx]}
        </motion.p>
      </AnimatePresence>

      {/* Progress dots */}
      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            style={{
              width: 7, height: 7, borderRadius: '50%',
              background: 'var(--brand-blue)',
            }}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
    </div>
  );
}
