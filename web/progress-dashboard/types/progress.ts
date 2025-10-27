export interface ProgressData {
  status?: string | null;
  train_loss?: number | null;
  val_loss?: number | null;
  val_acc1?: number | null;
  current_epoch?: number | null;
  total_epochs?: number | null;
  elapsed_sec?: number | null;
  message?: string | null;
  cpu_percent?: number | null;
  ram_percent?: number | null;
  gpu_percent?: number | null;
  gpu_memory_used?: number | null;
  gpu_memory_total?: number | null;
}
