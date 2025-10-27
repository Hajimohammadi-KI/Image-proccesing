'use client';

import { useEffect, useMemo, useState } from 'react';
import { GaugeCard, StatCard } from '../components/StatCard';
import type { ProgressData } from '../types/progress';
import type { SystemStats } from '../types/system';

const POLL_INTERVAL_MS = 1000;

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
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [systemError, setSystemError] = useState<string | null>(null);
  const [timestamp, setTimestamp] = useState<Date | null>(null);

  useEffect(() => {
    let isMounted = true;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    const fetchProgress = async () => {
      const progressRequest = fetch(`/progress.json?ts=${Date.now()}`, {
        cache: 'no-store'
      });
      const systemRequest = fetch(`/api/system?ts=${Date.now()}`, {
        cache: 'no-store'
      });

      try {
        const [progressResponse, systemResponse] = await Promise.allSettled([
          progressRequest,
          systemRequest
        ]);

        if (!isMounted) {
          return;
        }

        if (progressResponse.status === 'fulfilled' && progressResponse.value.ok) {
          const payload = (await progressResponse.value.json()) as ProgressData;
          setData(payload);
          setTimestamp(new Date());
          setError(null);
        } else {
          const reason =
            progressResponse.status === 'rejected'
              ? progressResponse.reason
              : new Error('Progress file not available yet');
          const message = reason instanceof Error ? reason.message : 'Failed to fetch progress';
          setError(message);
        }

        if (systemResponse.status === 'fulfilled' && systemResponse.value.ok) {
          const payload = (await systemResponse.value.json()) as SystemStats;
          setSystemStats(payload);
          setSystemError(null);
        } else {
          const reason =
            systemResponse.status === 'rejected'
              ? systemResponse.reason
              : new Error('System stats unavailable');
          const message = reason instanceof Error ? reason.message : 'Failed to fetch system stats';
          setSystemError(message);
        }
      } catch (err) {
        if (!isMounted) {
          return;
        }
        const message = err instanceof Error ? err.message : 'Failed to fetch dashboard data';
        setError(message);
        setSystemError(message);
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

  const cpuPercent = useMemo(
    () => sanitizePercent(systemStats?.cpu_percent ?? data?.cpu_percent),
    [systemStats?.cpu_percent, data?.cpu_percent]
  );
  const ramPercent = useMemo(
    () => sanitizePercent(systemStats?.ram_percent ?? data?.ram_percent),
    [systemStats?.ram_percent, data?.ram_percent]
  );
  const gpuPercent = useMemo(
    () => sanitizePercent(systemStats?.gpu_percent ?? data?.gpu_percent),
    [systemStats?.gpu_percent, data?.gpu_percent]
  );

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
            subtitle={formatPercent(systemStats?.cpu_percent ?? data?.cpu_percent)}
          />
          <GaugeCard
            label="RAM Engagement"
            percent={ramPercent}
            subtitle={formatPercent(systemStats?.ram_percent ?? data?.ram_percent)}
          />
          <GaugeCard
            label="GPU Engagement"
            percent={gpuPercent}
            subtitle={formatPercent(systemStats?.gpu_percent ?? data?.gpu_percent)}
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
            value={formatMemory(
              systemStats?.gpu_memory_used ?? data?.gpu_memory_used,
              systemStats?.gpu_memory_total ?? data?.gpu_memory_total
            )}
          />
        </div>

        {data?.message && <div className="statusText">{data.message}</div>}

        <div className="timestamp">
          Updated: {timestamp ? timestamp.toLocaleTimeString() : '—'}
        </div>
        {systemError && <div className="statusText">System stats: {systemError}</div>}
      </div>
    </main>
  );
}
