import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

import { NextResponse } from 'next/server';
import si from 'systeminformation';

interface SystemStatsResponse {
  cpu_percent: number | null;
  ram_percent: number | null;
  gpu_percent: number | null;
  gpu_memory_used: number | null;
  gpu_memory_total: number | null;
}

export async function GET(): Promise<NextResponse<SystemStatsResponse | { error: string }>> {
  try {
    const [cpuLoad, memory, graphics, nvidiaMetrics] = await Promise.all([
      si.currentLoad(),
      si.mem(),
      si.graphics(),
      readNvidiaMetrics(),
    ]);

    const controller = graphics.controllers?.find((item) => item.vendor?.toLowerCase().includes('nvidia'))
      ?? graphics.controllers?.[0];

    const cpuPercent = Number.isFinite(cpuLoad.currentLoad) ? cpuLoad.currentLoad : null;
    const ramSource = typeof memory.active === 'number' && memory.active > 0 ? memory.active : memory.used;
    const ramPercent = memory.total > 0 && typeof ramSource === 'number'
      ? (ramSource / memory.total) * 100
      : null;

  const rawGpuPercent = controller?.utilizationGpu ?? nvidiaMetrics?.gpuUtilization;
  const rawGpuMemUsed = controller?.memoryUsed ?? nvidiaMetrics?.memoryUsed;
  const rawGpuMemTotal = controller?.memoryTotal ?? nvidiaMetrics?.memoryTotal;

    const gpuPercent = typeof rawGpuPercent === 'number' && Number.isFinite(rawGpuPercent)
      ? rawGpuPercent
      : null;
    const gpuMemoryUsed = typeof rawGpuMemUsed === 'number' && Number.isFinite(rawGpuMemUsed)
      ? rawGpuMemUsed
      : null;
    const gpuMemoryTotal = typeof rawGpuMemTotal === 'number' && Number.isFinite(rawGpuMemTotal)
      ? rawGpuMemTotal
      : null;

    return NextResponse.json({
      cpu_percent: cpuPercent,
      ram_percent: ramPercent !== null ? Math.min(Math.max(ramPercent, 0), 100) : null,
  gpu_percent: gpuPercent !== null ? Math.min(Math.max(gpuPercent, 0), 100) : null,
  gpu_memory_used: gpuMemoryUsed,
  gpu_memory_total: gpuMemoryTotal,
    });
  } catch (error) {
    console.error('Failed to collect system stats', error);
    return NextResponse.json({ error: 'Failed to collect system stats' }, { status: 500 });
  }
}

const execFileAsync = promisify(execFile);

async function readNvidiaMetrics(): Promise<
  | {
      gpuUtilization: number;
      memoryUsed: number;
      memoryTotal: number;
    }
  | null
> {
  try {
    const { stdout } = await execFileAsync('nvidia-smi', [
      '--query-gpu=utilization.gpu,memory.used,memory.total',
      '--format=csv,noheader,nounits',
    ], {
      windowsHide: true,
      timeout: 1000,
    });

    const line = stdout.trim().split('\n')[0];
    if (!line) {
      return null;
    }
    const [utilStr, memUsedStr, memTotalStr] = line.split(',').map((part) => part.trim());
    const gpuUtilization = Number.parseFloat(utilStr);
    const memoryUsed = Number.parseFloat(memUsedStr);
    const memoryTotal = Number.parseFloat(memTotalStr);

    if ([gpuUtilization, memoryUsed, memoryTotal].some((value) => Number.isNaN(value))) {
      return null;
    }

    return { gpuUtilization, memoryUsed, memoryTotal };
  } catch (error) {
    // nvidia-smi not found or timed out. Return null so callers can fall back.
    return null;
  }
}
