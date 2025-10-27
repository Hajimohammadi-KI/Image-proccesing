import type { CSSProperties } from 'react';

interface StatCardProps {
  label: string;
  value: string;
}

interface GaugeCardProps {
  label: string;
  percent: number | null;
  subtitle?: string;
}

export function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="statCard">
      <div className="statLabel">{label}</div>
      <div className="statValue">{value}</div>
    </div>
  );
}

export function GaugeCard({ label, percent, subtitle }: GaugeCardProps) {
  const hasValue = percent !== null && !Number.isNaN(percent);
  const value = hasValue ? Math.round(percent!) : 0;
  const gaugeStyle: CSSProperties = {
    background: `conic-gradient(var(--accent) ${value}%, rgba(0, 0, 0, 0.08) ${value}% 100%)`
  };

  return (
    <div className="gaugeCard">
      <div className="gaugeShell" style={gaugeStyle}>
        <div className="gaugeCore">
          <span className="gaugeValue">{hasValue ? `${value}%` : '—'}</span>
        </div>
      </div>
      <div className="statLabel">{label}</div>
      {subtitle ? <div className="gaugeSubtitle">{subtitle}</div> : null}
    </div>
  );
}
