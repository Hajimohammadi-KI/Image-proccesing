'use client';

import { useEffect, useMemo, useState } from 'react';
import { GaugeCard, StatCard } from '../components/StatCard';
import type { ProgressData } from '../types/progress';

const POLL_INTERVAL_MS = 5000;

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-';
  }
  return Number(value).toFixed(digits);
}

function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-';
  }
  return `${Number(value).toFixed(digits)}%`;
}

function sanitizePercent(value: number | null | undefined): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return Math.min(100, Math.max(0, numeric));
}

function formatMemory(
  used: number | null | undefined,
  total: number | null | undefined
): string {
  if (
    used === null ||
    used === undefined ||
    Number.isNaN(used) ||
    total === null ||
    total === undefined ||
    Number.isNaN(total)
  ) {
    return '-';
  }
  return `${used.toFixed(0)} / ${total.toFixed(0)} MB`;
}

function formatElapsed(seconds: number | null | undefined): string {
  if (!seconds || Number.isNaN(seconds)) {
    return '-';
  }
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

function getProgressPercent(data: ProgressData | null): number {
  if (!data || !data.total_epochs || !data.current_epoch) {
    return 0;
  }
  if (data.total_epochs <= 0) {
    return 0;
  }
  const ratio = (data.current_epoch / data.total_epochs) * 100;
  return Math.min(100, Math.max(0, Math.round(ratio)));
}

export default function DashboardPage() {
  const [data, setData] = useState<ProgressData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [timestamp, setTimestamp] = useState<Date | null>(null);

  useEffect(() => {
    let isMounted = true;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    const fetchProgress = async () => {
      try {
        const response = await fetch(`/progress.json?ts=${Date.now()}`, {
          cache: 'no-store'
        });

        if (!response.ok) {
          throw new Error('Progress file not available yet');
        }

        const payload = (await response.json()) as ProgressData;

        if (!isMounted) {
          return;
        }

        setData(payload);
        setTimestamp(new Date());
        setError(null);
      } catch (err) {
        if (!isMounted) {
          return;
        }
        const message = err instanceof Error ? err.message : 'Failed to fetch progress';
        setError(message);
      }
    };

    fetchProgress();
    intervalId = setInterval(fetchProgress, POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []);

  const percent = useMemo(() => getProgressPercent(data), [data]);

  const cpuPercent = useMemo(() => sanitizePercent(data?.cpu_percent), [data?.cpu_percent]);
  const ramPercent = useMemo(() => sanitizePercent(data?.ram_percent), [data?.ram_percent]);
  const gpuPercent = useMemo(() => sanitizePercent(data?.gpu_percent), [data?.gpu_percent]);

  const progressLabel = useMemo(() => {
    if (!data || !data.total_epochs || !data.current_epoch) {
      return 'Waiting for data...';
    }
    return `${percent}% (${data.current_epoch}/${data.total_epochs})`;
  }, [data, percent]);

  const status = data?.status ?? 'unknown';

  return (
    <main>
      <div className="card">
        <h1>Training Progress</h1>

        {error && <div className="errorBanner">{error}</div>}

        <div className="progressWrapper">
          <div className="progressBar" style={{ width: `${percent}%` }} />
        </div>
        <div className="progressText">{progressLabel}</div>

        <div className="gaugeGrid">
          <GaugeCard
            label="CPU Engagement"
            percent={cpuPercent}
            subtitle={formatPercent(data?.cpu_percent)}
          />
          <GaugeCard
            label="RAM Engagement"
            percent={ramPercent}
            subtitle={formatPercent(data?.ram_percent)}
          />
          <GaugeCard
            label="GPU Engagement"
            percent={gpuPercent}
            subtitle={formatPercent(data?.gpu_percent)}
          />
        </div>

        <div className="statsGrid">
          <StatCard label="Status" value={status} />
          <StatCard label="Train Loss" value={formatNumber(data?.train_loss)} />
          <StatCard label="Val Loss" value={formatNumber(data?.val_loss)} />
          <StatCard label="Val Acc@1" value={formatNumber(data?.val_acc1, 4)} />
          <StatCard label="Elapsed" value={formatElapsed(data?.elapsed_sec)} />
          <StatCard
            label="GPU Memory"
            value={formatMemory(data?.gpu_memory_used, data?.gpu_memory_total)}
          />
        </div>

        {data?.message && <div className="statusText">{data.message}</div>}

        <div className="timestamp">
          Updated: {timestamp ? timestamp.toLocaleTimeString() : '—'}
        </div>
      </div>
    </main>
  );
}
