import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function HeatmapViewer({ originalSrc, heatmapSrc, showHeatmap }) {
  const displaySrc = showHeatmap && heatmapSrc ? heatmapSrc : originalSrc;

  return (
    <div style={{ position: 'relative', width: '100%', background: '#0a0a0a' }}>
      <AnimatePresence mode="wait">
        <motion.img
          key={showHeatmap ? 'heatmap' : 'original'}
          src={displaySrc}
          alt={showHeatmap ? 'Grad-CAM Heatmap' : 'Chest X-Ray'}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          style={{
            width: '100%',
            maxHeight: '360px',
            objectFit: 'contain',
            display: 'block',
          }}
        />
      </AnimatePresence>

      {/* Colormap Legend (shown when heatmap is visible) */}
      {showHeatmap && heatmapSrc && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            background: 'rgba(0,0,0,0.7)',
            borderRadius: 6,
            padding: '4px 10px',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span style={{ fontSize: 10, color: '#9CA3AF' }}>Low</span>
          <div style={{
            width: 80, height: 8,
            borderRadius: 4,
            background: 'linear-gradient(90deg, #00F, #0FF, #0F0, #FF0, #F00)',
          }} />
          <span style={{ fontSize: 10, color: '#9CA3AF' }}>High</span>
        </motion.div>
      )}
    </div>
  );
}
