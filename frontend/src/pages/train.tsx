import { useEffect, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  CheckCircle2,
  CircleDashed,
  Download,
  FolderCheck,
  LoaderCircle,
  Play,
  XCircle,
} from 'lucide-react'

import { ApiError, api } from '@/lib/api'
import { fmtNum } from '@/lib/format'
import type {
  JobStatusResponse,
  StartJobResponse,
  TrainConfig,
  TrainFileCheck,
  TrainResult,
  VerifiedDataset,
} from '@/lib/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { StatTile, VizCard, axisTick, gridProps } from '@/components/viz'
import { cn } from '@/lib/utils'

const MODELS = [
  { id: 'xgboost', label: 'XGBoost (trees)', hint: '~1 min on this machine (full dataset)' },
  { id: 'lstm', label: 'LSTM', hint: '~10 min GPU / longer CPU (15 epochs)' },
  { id: 'gru', label: 'GRU', hint: '~10 min GPU / longer CPU (15 epochs)' },
  { id: 'transformer', label: 'Transformer', hint: '~10-20 min GPU (15 epochs)' },
] as const

const STAGES = [
  { name: 'verify', label: 'Verify schema' },
  { name: 'prepare', label: 'Clean & prepare' },
  { name: 'baseline', label: 'Persistence baseline' },
  { name: 'train', label: 'Train model' },
  { name: 'evaluate', label: 'Evaluate' },
] as const

/** Poll a running job every 1.5 s until it leaves `running`. */
function useJob(jobId: string | null) {
  const [status, setStatus] = useState<JobStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const tick = async () => {
      try {
        const s = await api.jobStatus(jobId)
        if (!alive) return
        setStatus(s)
        setError(null)
        if (s.status === 'running') timer = setTimeout(tick, 1500)
      } catch (e) {
        if (!alive) return
        setError((e as Error).message)
        timer = setTimeout(tick, 3000) // transient network hiccup — keep trying
      }
    }
    setStatus(null)
    tick()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [jobId])

  return { status, error }
}

/** Per-file checklist survives failures too — the 422 body carries it. */
function filesFromError(e: unknown): TrainFileCheck[] | null {
  if (e instanceof ApiError && Array.isArray(e.detail)) {
    const arr = e.detail as unknown[]
    return arr.every((f) => f && typeof f === 'object' && 'name' in (f as object))
      ? (arr as TrainFileCheck[])
      : null
  }
  return null
}

export default function Train() {
  // dataset ingestion
  const [tab, setTab] = useState<'path' | 'upload'>('path')
  const [pathInput, setPathInput] = useState('E:\\Solar_gemini\\unisolar')
  const [picked, setPicked] = useState<File[]>([])
  const [dsBusy, setDsBusy] = useState(false)
  const [dsError, setDsError] = useState<string | null>(null)
  const [checked, setChecked] = useState<TrainFileCheck[] | null>(null)
  const [dataset, setDataset] = useState<VerifiedDataset | null>(null)

  // configure & run
  const [config, setConfig] = useState<TrainConfig | null>(null)
  const [model, setModel] = useState<string>('xgboost')
  const [fastTest, setFastTest] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [startError, setStartError] = useState<string | null>(null)
  const { status, error: pollError } = useJob(jobId)

  // auto-scroll log pane
  const logRef = useRef<HTMLPreElement | null>(null)
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [status?.log_tail.length])

  useEffect(() => {
    api.trainConfig().then(setConfig).catch(() => setConfig(null))
  }, [])

  const verify = async () => {
    setDsBusy(true)
    setDsError(null)
    setChecked(null)
    try {
      const ds =
        tab === 'path'
          ? await api.verifyPathDataset(pathInput)
          : await api.uploadDataset(picked)
      setDataset(ds)
      setChecked(ds.files)
    } catch (e) {
      setDataset(null)
      setDsError(e instanceof Error ? e.message : String(e))
      setChecked(filesFromError(e))
    } finally {
      setDsBusy(false)
    }
  }

  const start = async () => {
    if (!dataset) return
    setStartError(null)
    try {
      const res: StartJobResponse = await api.startJob({
        dataset_id: dataset.dataset_id,
        model,
        fast_test: fastTest,
      })
      setJobId(res.job_id)
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e))
    }
  }

  const running = status?.status === 'running'
  const result = status?.result ?? null
  const failed = status?.status === 'failed'

  return (
    <div className="space-y-4">
      {/* ---- 1 · dataset ------------------------------------------------ */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">1 · Dataset folder</CardTitle>
          <CardDescription>
            Point at a UNISOLAR-style folder (the three source CSVs) on the
            server, or upload them from this machine. Verification is read-only;
            nothing outside this job's directory is touched.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Tabs
            value={tab}
            onValueChange={(v) => {
              setTab(v as 'path' | 'upload')
              setChecked(null)
              setDsError(null)
            }}
          >
            <TabsList>
              <TabsTrigger value="path">Server path</TabsTrigger>
              <TabsTrigger value="upload">Upload CSVs</TabsTrigger>
            </TabsList>
          </Tabs>

          {tab === 'path' ? (
            <div className="flex items-end gap-2">
              <div className="min-w-72 flex-1 space-y-1.5">
                <Label htmlFor="train-path" className="text-xs">
                  Folder containing the three UNISOLAR CSVs
                </Label>
                <input
                  id="train-path"
                  type="text"
                  spellCheck={false}
                  className="h-8 w-full rounded-md border bg-background px-2 font-mono text-xs"
                  placeholder="E:\path\to\unisolar"
                  value={pathInput}
                  onChange={(e) => setPathInput(e.target.value)}
                />
              </div>
              <Button size="sm" onClick={verify} disabled={dsBusy || !pathInput.trim()}>
                {dsBusy ? <LoaderCircle className="animate-spin" /> : <FolderCheck />}
                Verify folder
              </Button>
            </div>
          ) : (
            <div className="flex items-end gap-2">
              <div className="min-w-72 flex-1 space-y-1.5">
                <Label htmlFor="train-files" className="text-xs">
                  Solar_Energy_Generation.csv · Weather_Data_reordered_all.csv · Solar_Site_Details.csv
                </Label>
                <input
                  id="train-files"
                  type="file"
                  multiple
                  accept=".csv"
                  className="block w-full text-xs text-muted-foreground file:mr-2 file:rounded-md file:border file:bg-background file:px-2 file:py-1 file:text-xs"
                  onChange={(e) => setPicked(Array.from(e.target.files ?? []))}
                />
              </div>
              <Button size="sm" onClick={verify} disabled={dsBusy || picked.length === 0}>
                {dsBusy ? <LoaderCircle className="animate-spin" /> : <FolderCheck />}
                Upload &amp; verify
              </Button>
            </div>
          )}

          {dsError && (
            <p className="text-xs text-status-bad-text" role="alert">{dsError}</p>
          )}

          {checked && (
            <div className="overflow-hidden rounded-lg border">
              <table className="w-full text-xs">
                <thead className="bg-muted/50 text-left text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 font-medium">File</th>
                    <th className="px-3 py-2 font-medium">Rows</th>
                    <th className="px-3 py-2 font-medium">Check</th>
                  </tr>
                </thead>
                <tbody>
                  {checked.map((f) => (
                    <tr key={f.name} className="border-t">
                      <td className="px-3 py-2 font-mono">{f.name}</td>
                      <td className="px-3 py-2 font-mono tabular-nums">
                        {f.rows === null ? '—' : f.rows.toLocaleString()}
                      </td>
                      <td className="px-3 py-2">
                        {f.ok ? (
                          <span className="inline-flex items-center gap-1 text-status-good-text">
                            <CheckCircle2 className="size-3.5" aria-hidden /> {f.detail}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-status-bad-text">
                            <XCircle className="size-3.5" aria-hidden /> {f.detail}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {dataset && (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
              <StatTile label="Generation rows"
                        value={dataset.profile.generation_rows.toLocaleString()} />
              <StatTile label="Sites" value={String(dataset.profile.sites)} />
              <StatTile label="Campuses" value={String(dataset.profile.campuses)} />
              <StatTile label="Cadence"
                        value={dataset.profile.cadence_minutes !== null
                          ? `${dataset.profile.cadence_minutes} min` : '—'} />
              <StatTile label="Span"
                        value={`${dataset.profile.start.slice(0, 10)} → ${dataset.profile.end.slice(0, 10)}`} />
              <StatTile label="Target missing"
                        value={`${fmtNum(dataset.profile.target_missing_pct, 2)}%`} />
            </div>
          )}
        </CardContent>
      </Card>

      {/* ---- 2 · configure ---------------------------------------------- */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">2 · Configure &amp; train</CardTitle>
          <CardDescription>
            Hyperparameters are the same ones that produced the served v1 models
            (configs/*.yaml, read-only). One heavy job runs at a time.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Model</Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="w-52 font-mono text-xs" aria-label="Model to train">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODELS.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-[11px] text-muted-foreground">
                {MODELS.find((m) => m.id === model)?.hint}
              </p>
            </div>

            <label className="flex cursor-pointer items-center gap-2 pb-1 text-xs text-muted-foreground">
              <input
                type="checkbox"
                className="size-3.5 accent-[var(--primary)]"
                checked={fastTest}
                onChange={(e) => setFastTest(e.target.checked)}
              />
              fast test mode (tiny model, seconds — smoke test only)
            </label>

            <div className="ml-auto">
              <Button onClick={start} disabled={!dataset || !!running}>
                {running ? <LoaderCircle className="animate-spin" /> : <Play />}
                {running ? 'Training…' : 'Start training'}
              </Button>
            </div>
          </div>

          {(startError || pollError) && (
            <p className="text-xs text-status-bad-text" role="alert">
              {startError ?? pollError}
            </p>
          )}

          {config && (
            <div className="rounded-lg border bg-muted/30 px-3 py-2">
              <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                config snapshot ({model})
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
                {Object.entries(config.training).map(([k, v]) => (
                  <span key={k} className="text-chart-1">
                    <span className="opacity-70">{k}</span>={String(v)}
                  </span>
                ))}
                {Object.entries(config.models[model]?.params ?? {}).map(([k, v]) => (
                  <span key={k} className="text-chart-2">
                    <span className="opacity-70">{k}</span>={String(v)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ---- 3 · progress ------------------------------------------------ */}
      {(jobId || running) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">3 · Progress</CardTitle>
            <CardDescription className="font-mono">
              job {jobId}
              {status?.elapsed_s !== undefined && ` · ${formatElapsed(status.elapsed_s)}`}
              {failed && status?.error && ` · ${status.error}`}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <ol className="flex flex-wrap gap-x-5 gap-y-2">
              {STAGES.map((s) => {
                const st = status?.stages.find((x) => x.name === s.name)?.status
                return (
                  <li key={s.name} className="flex items-center gap-2 text-xs">
                    <StageMark state={st} />
                    <span className={cn(st === 'done' ? 'text-foreground' : 'text-muted-foreground')}>
                      {s.label}
                    </span>
                  </li>
                )
              })}
            </ol>

            <pre
              ref={logRef}
              className="max-h-64 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-[11px] leading-4"
              aria-label="Training log"
            >
              {(status?.log_tail ?? []).join('\n') || 'waiting for output…'}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* ---- 4 · results -------------------------------------------------- */}
      {result && !failed && <ResultsCard result={result} jobId={jobId!} />}
    </div>
  )
}

/* ---- results ----------------------------------------------------------- */

function ResultsCard({ result, jobId }: { result: TrainResult; jobId: string }) {
  const m = result.model
  const base = result.persistence.test_all
  const t = result.test_all
  const delta = t.mae !== null && base.mae !== null ? t.mae - base.mae : null

  const perSite: { site: string; mae: number }[] = result.metrics_per_site
    .filter((r) => r.scope === 'SITE' && r.split === 'test')
    .sort((a, b) => (b.mae ?? 0) - (a.mae ?? 0))
    .slice(0, 14)
    .map((r) => ({ site: `site ${r.site_id}`, mae: r.mae ?? 0 }))
  const perSiteRows = perSite.map((r) => [r.site, fmtNum(r.mae, 3)])

  const history = (m.training_history ?? []).map((h) => ({
    epoch: h.epoch,
    val_rmse: h.val_rmse ?? null,
  }))
  const importance = (m.feature_importance_top20 ?? [])
    .slice(0, 14)
    .map((r) => ({ feature: r.feature, gain: r.gain }))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">4 · Results — {m.model_name}</CardTitle>
        <CardDescription>
          Test split, ALL scope · nRMSE denominator = train-period observed range
          (D-011) · finished {result.generated_at.slice(0, 16).replace('T', ' ')} UTC
          {result.config_used.fast_test && ' · FAST-TEST run — tiny model, numbers are not meaningful'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <StatTile
            label="Test MAE"
            value={`${fmtNum(t.mae, 3)} kW`}
            sub={
              delta === null ? undefined : (
                <span className={delta <= 0 ? 'text-status-good-text' : 'text-status-bad-text'}>
                  {delta <= 0 ? '−' : '+'}
                  {fmtNum(Math.abs(delta), 3)} kW vs persistence ({fmtNum(base.mae, 3)})
                </span>
              )
            }
          />
          <StatTile label="Test RMSE" value={`${fmtNum(t.rmse, 3)} kW`}
                    sub={`nRMSE ${fmtNum(t.nrmse, 3)}`} />
          <StatTile label="Test R²" value={fmtNum(t.r2, 4)}
                    sub={`n = ${t.n_eval.toLocaleString()}`} />
          <StatTile label="Daylight MAE" value={`${fmtNum(t.daylight_mae, 3)} kW`}
                    sub={`daylight nRMSE ${fmtNum(t.daylight_nrmse, 3)}`} />
          <StatTile label="Fit time" value={`${fmtNum(m.fit_seconds, 1)} s`}
                    sub={`pipeline total ${fmtNum(result.timing.total_s, 1)} s${m.device ? ` · ${m.device}` : ''}`} />
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <VizCard
            title="Worst sites — test MAE"
            description="Top sites by mean absolute error on the held-out test window."
            table={<MiniTable head={['Site', 'MAE (kW)']} rows={perSiteRows} />}
          >
            <figure className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={perSite} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="site" tick={{ ...axisTick }} tickLine={false}
                         axisLine={{ stroke: 'var(--viz-axis)' }} interval={0} angle={-35}
                         textAnchor="end" height={54} />
                  <YAxis width={44} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
                  <RTooltip content={<TooltipKw />} />
                  <Bar dataKey="mae" name="test MAE" fill="var(--chart-1)"
                       radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </figure>
          </VizCard>

          {history.length > 0 ? (
            <VizCard
              title="Training history"
              description="Validation RMSE (normalized units) per epoch — early stopping restored the best checkpoint."
              table={
                <MiniTable head={['Epoch', 'Val RMSE']}
                           rows={history.map((h) => [String(h.epoch), fmtNum(h.val_rmse, 4)])} />
              }
            >
              <figure className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="epoch" tick={{ ...axisTick }} tickLine={false} axisLine={false} />
                    <YAxis width={48} tick={{ ...axisTick }} tickLine={false} axisLine={false}
                           domain={['auto', 'auto']} />
                    <RTooltip content={<TooltipKw name="val RMSE" digits={4} suffix="" />} />
                    <Line type="monotone" dataKey="val_rmse" name="val RMSE"
                          stroke="var(--chart-2)" strokeWidth={2} dot={false}
                          isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </figure>
            </VizCard>
          ) : (
            <VizCard
              title="Feature importance (gain)"
              description="Top features by accumulated gain — same signal the Explainability page SHAP-analyzes for v1."
              table={
                <MiniTable head={['Feature', 'Gain']}
                           rows={importance.map((r) => [r.feature, fmtNum(r.gain, 4)])} />
              }
            >
              <figure className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={importance} layout="vertical"
                            margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid {...gridProps} horizontal={false} vertical />
                    <XAxis type="number" tick={{ ...axisTick }} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="feature" width={170}
                           tick={{ ...axisTick, fontSize: 10 }} tickLine={false} axisLine={false} />
                    <RTooltip content={<TooltipKw name="gain" suffix="" />} />
                    <Bar dataKey="gain" name="gain" fill="var(--chart-3)"
                         radius={[0, 3, 3, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </figure>
            </VizCard>
          )}
        </div>

        {/* provenance strip */}
        <div className="grid gap-3 text-xs md:grid-cols-3">
          <ProvenanceBlock title="Split (D-011 chronological)">
            {(['train', 'val', 'test'] as const).map((k) => (
              <div key={k} className="font-mono text-[11px]">
                {k}: {result.split[k].rows.toLocaleString()} rows ·{' '}
                {result.split[k].start.slice(0, 10)} → {result.split[k].end.slice(0, 10)}
              </div>
            ))}
          </ProvenanceBlock>
          <ProvenanceBlock title="Prepare">
            <div className="font-mono text-[11px]">
              merged {result.cleaning_and_prepare.merged_rows.toLocaleString()} rows →{' '}
              {result.cleaning_and_prepare.features_cols} feature cols
              <br />
              tz {result.cleaning_and_prepare.timezone_chosen} ·{' '}
              {result.cleaning_and_prepare.cleaning_ops.length} cleaning operations logged
            </div>
          </ProvenanceBlock>
          <ProvenanceBlock title="Artifacts (job-scoped)">
            <div className="flex flex-col gap-1">
              {(['metrics.csv', 'predictions_test.parquet', 'result.json'] as const).map((name) => (
                <a key={name}
                   className="inline-flex items-center gap-1 underline underline-offset-2 hover:text-foreground"
                   href={`/api/v1/train/jobs/${jobId}/artifacts/${name}`}>
                  <Download className="size-3" aria-hidden /> {name}
                </a>
              ))}
            </div>
          </ProvenanceBlock>
        </div>

        <p className="text-[11px] text-muted-foreground">
          Display-only run (D-024): artifacts live under
          {' '}<code>data/train_jobs/…/{jobId}/</code> — the served v1 models and recorded
          RESULTS.md numbers were never modified.
        </p>
      </CardContent>
    </Card>
  )
}

function ProvenanceBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border bg-muted/30 px-3 py-2">
      <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">{title}</div>
      {children}
    </div>
  )
}

function MiniTable({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="max-h-72 overflow-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-card text-left text-muted-foreground">
          <tr>{head.map((h) => <th key={h} className="px-2 py-1 font-medium">{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t">
              {r.map((c, j) => (
                <td key={j} className={cn('px-2 py-1', j > 0 && 'text-right font-mono tabular-nums')}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function StageMark({ state }: { state?: 'running' | 'done' }) {
  if (state === 'done')
    return <CheckCircle2 className="size-4 text-status-good-text" aria-label="done" />
  if (state === 'running')
    return <LoaderCircle className="size-4 animate-spin text-primary" aria-label="running" />
  return <CircleDashed className="size-4 text-muted-foreground/50" aria-label="pending" />
}

/** Shared metric tooltip — mono value column. */
function TooltipKw({
  active,
  payload,
  label,
  name,
  suffix = ' kW',
  digits = 3,
}: {
  active?: boolean
  payload?: { name?: string; value?: number; color?: string }[]
  label?: string
  name?: string
  suffix?: string
  digits?: number
}) {
  if (!active || !payload?.length) return null
  const p = payload[0]
  return (
    <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
      <div className="mb-1 font-mono text-muted-foreground">{label}</div>
      <div className="flex items-center gap-2 leading-5">
        <span className="inline-block size-2 rounded-full" style={{ background: p.color }} />
        <span className="text-muted-foreground">{name ?? p.name}</span>
        <span className="ml-auto pl-4 font-mono tabular-nums">
          {typeof p.value === 'number' ? `${fmtNum(p.value, digits)}${suffix}` : '—'}
        </span>
      </div>
    </div>
  )
}

function formatElapsed(s: number): string {
  return s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`
}
