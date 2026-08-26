/** Shapes returned by /api/v1 (PRD §33-36). Timestamps are naive Melbourne
 * wall time (D-007) — render as strings, never through Date timezone munging. */

export interface Site {
  site_id: number
  campus_id: number
  latitude: number | null
  longitude: number | null
}

export interface DatasetInfo {
  n_rows: number
  n_sites: number
  n_campuses: number
  cadence_minutes: number
  timezone: string
  n_features_engineered: number | null
}

export interface ModelEntry {
  model_id: string
  family: string
  artifact: string | null
  served: boolean
}

export interface Metrics {
  model_id: string
  split: string
  scope: string
  mae: number
  rmse: number
  r2: number
  nrmse: number | null
  daylight_mae: number | null
  daylight_nrmse: number | null
  n_eval: number
}

export interface HistoryRow {
  timestamp: string
  campus_id?: number
  power: number | null
  is_daylight?: boolean
  temperature?: number | null
  humidity?: number | null
  wind_speed?: number | null
}

export interface HistoryResponse {
  site_id: number
  resolution: string
  n_rows: number
  rows: HistoryRow[]
}

export type Regime = 'day_lag' | 'day_nolag' | 'night_lag' | 'night_nolag'

export interface PredictionPoint {
  timestamp: string
  prediction: number | null
  lower_bound?: number
  upper_bound?: number
  confidence_level?: number
  regime?: Regime
}

export interface ForecastResponse {
  site_id: number
  model: string
  forecast_horizon: number
  predictions: PredictionPoint[]
}

/* ---- bundled snapshot JSONs (scripts/export_frontend_data.py) ---- */

export interface ShapFeature {
  rank: number
  feature: string
  mean_abs_shap: number
  share: number
}

export interface ShapBundle {
  explained_run: string
  sample_rows: number
  additivity_max_abs_err: number
  features: ShapFeature[]
}

export interface SiteSummary {
  site_id: number
  campus_id: number
  latitude: number | null
  longitude: number | null
  capacity_kwp: number | null
  first_ts: string
  last_ts: string
  n_rows: number
  expected_slots: number
  row_availability_pct: number
  power_observed_pct: number
  mean_daylight_kwh: number | null
  max_daylight_kwh: number | null
}

export interface QualityBundle {
  generation: {
    duplicate_keys: number
    impossible_values: number
    missing_slots: number
    groups_checked: number
    worst_sites: { site_id: number; missing_pct: number }[]
    power_missing_raw_pct: number
  }
  weather_overall_pct: Record<string, number>
  weather_monthly: {
    month: string
    temperature_pct: number | null
    humidity_pct: number | null
    wind_speed_pct: number | null
  }[]
  cleaning: { total_operations: number; weather_interpolated_rows: number }
  outliers_flagged: number
}

export interface SiteMonthlyBundle {
  months: string[]
  sites: { site_id: number; campus_id: number; monthly_kwh: (number | null)[] }[]
  campuses: { campus_id: number; n_sites: number; monthly_kwh_mean: (number | null)[] }[]
}

export interface QualityExtraBundle {
  hist_bin_edges_kw: number[]
  hist_bin_labels: string[]
  hist_by_site: { site_id: number; counts: number[] }[]
  hist_total: number[]
  availability: {
    site_id: number
    row_availability_pct: number
    daylight_power_obs_pct: number
  }[]
}

/* ---- Train page (job API, D-024/D-025) ---- */

export interface TrainFileCheck {
  name: string
  ok: boolean
  rows: number | null
  detail: string
}

export interface DatasetProfile {
  generation_rows: number
  weather_rows: number
  sites: number
  campuses: number
  start: string
  end: string
  cadence_minutes: number | null
  target_missing_pct: number | null
}

/** POST /train/datasets/{path,upload} response. */
export interface VerifiedDataset {
  dataset_id: string
  mode: string
  raw_dir: string
  files: TrainFileCheck[]
  profile: DatasetProfile
}

export type StageName = 'verify' | 'prepare' | 'baseline' | 'train' | 'evaluate'

export interface TrainStage {
  name: string
  status: 'running' | 'done'
}

export type JobStatus = 'running' | 'done' | 'failed' | 'unknown'

export interface MetricAll {
  n_eval: number
  n_missing: number
  mae: number | null
  rmse: number | null
  r2: number | null
  nrmse: number | null
  daylight_n: number | null
  daylight_mae: number | null
  daylight_nrmse: number | null
}

export interface MetricRow extends MetricAll {
  split: 'val' | 'test'
  scope: 'ALL' | 'SITE'
  site_id: number | ''
}

/** result.json written by scripts/train_from_folder.py — everything the
 * Results card shows. Numeric fields arrive as real JSON numbers (the
 * orchestrator sanitizes numpy scalars before dumping). */
export interface TrainResult {
  generated_at: string
  dataset: { files: TrainFileCheck[]; profile: DatasetProfile }
  cleaning_and_prepare: {
    merged_rows: number
    features_rows: number
    features_cols: number
    engineered_columns: number
    timezone_chosen: string
    timezone_candidates?: unknown
    cleaning_ops: { dataset: string; operation: string; rows_affected: number; detail: string }[]
    validation: unknown
  }
  split: Record<'train' | 'val' | 'test', { rows: number; observed_rows: number; start: string; end: string }>
  config_used: { ratios: [number, number, number]; seed: number; fast_test: boolean }
  timing: Record<string, number>
  persistence: { val_all: MetricAll; test_all: MetricAll }
  model: {
    model_name: string
    fit_seconds: number
    device?: string
    best_iteration?: number
    epochs_ran?: number
    best_val_rmse_normalized?: number
    training_history?: { epoch: number; lr?: number; val_rmse?: number }[]
    lookback_steps?: number
    channels?: string[]
    feature_importance_top20?: { feature: string; gain: number }[]
    params_used: Record<string, string | number | boolean>
    artifacts: Record<string, string>
  }
  metrics_per_site: MetricRow[]
  test_all: MetricAll
  val_all: MetricAll
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  stages: TrainStage[]
  stage: string
  elapsed_s?: number
  log_tail: string[]
  error: string | null
  result: TrainResult | null
}

export interface StartJobResponse {
  job_id: string
  model: string
  hint: string
  status: string
}

export interface TrainConfig {
  training: Record<string, string | number | boolean>
  models: Record<string, { params: Record<string, string | number | boolean> }>
}

/* ---- Post-PRD visual bundles (export_frontend_data.py) ---- */

export interface EdaProfilesBundle {
  hour_of_day: {
    slots: string[]
    mean_kw: Record<string, (number | null)[]> // "ALL" + campus ids
  }
  correlation: {
    campuses: number[]
    vars: string[]
    power_corr: (number | null)[][] // campuses × vars
  }
}

export interface MissingnessTimelineBundle {
  months: string[]
  generation_missing_slot_pct: number[]
}

export interface EvalSeriesMetrics {
  mae: number
  rmse: number
  r2: number
  nrmse: number
}

/** Hourly overlay for one run — index is the bundle-level shared `hourly_t`
 * (identical across runs, deduped at export for size). */
export interface HourlySeries {
  actual: (number | null)[]
  predicted: (number | null)[]
}

export interface ResidualHist {
  edges: number[] // 41 edges → 40 bins
  counts: number[]
}

export interface EvalSeriesEntry {
  metrics: EvalSeriesMetrics
  hourly_all?: HourlySeries
  daily_by_site?: Record<string, { actual: (number | null)[]; predicted: (number | null)[] }>
  scatter_sample?: { actual: number[]; predicted: number[] }
  residual_hist?: ResidualHist
}

export interface EvalSeriesBundle {
  test_window: { start: string; end: string }
  /** Shared hourly index for every run's `hourly_all` arrays. */
  hourly_t: string[]
  models: Record<string, EvalSeriesEntry>
}

export interface CrossSiteEntry {
  seen_val_all: EvalSeriesMetrics | null
  unseen_test_all: EvalSeriesMetrics | null
  unseen_site_r2: { site_id: number; r2: number }[]
}

export interface CrossSiteSummaryBundle {
  models: Record<string, CrossSiteEntry>
}
