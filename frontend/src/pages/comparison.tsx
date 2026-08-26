import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Download } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ChartTooltip, VizCard, axisTick, gridProps, legendStyle } from '@/components/viz'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { downloadCsv, fmtNum } from '@/lib/format'
import type { CrossSiteSummaryBundle, EvalSeriesBundle, Metrics } from '@/lib/types'
import evalJson from '@/data/evaluation_series.json'
import crossJson from '@/data/cross_site_summary.json'

const ev = evalJson as EvalSeriesBundle
const cs = crossJson as CrossSiteSummaryBundle

const RUN_ORDER = ['zero', 'mean_global', 'mean_site', 'persistence_prev_day',
                   'xgboost', 'lstm', 'gru', 'transformer']
const SERIES_RUNS = ['xgboost', 'lstm', 'gru', 'transformer']
const COMPACT_METRICS = [
  { key: 'mae', label: 'MAE (kWh)', lowerBetter: true },
  { key: 'rmse', label: 'RMSE (kWh)', lowerBetter: true },
  { key: 'r2', label: 'R²', lowerBetter: false },
  { key: 'nrmse', label: 'nRMSE', lowerBetter: true },
] as const
type CompactKey = (typeof COMPACT_METRICS)[number]['key']

const MODEL_ORDER = ['persistence', 'xgboost', 'lstm', 'gru', 'transformer']

const METRICS = [
  { key: 'mae', label: 'MAE (kWh)', lowerBetter: true },
  { key: 'rmse', label: 'RMSE (kWh)', lowerBetter: true },
  { key: 'r2', label: 'R²', lowerBetter: false },
  { key: 'nrmse', label: 'nRMSE', lowerBetter: true },
  { key: 'daylight_mae', label: 'Daylight MAE (kWh)', lowerBetter: true },
  { key: 'daylight_nrmse', label: 'Daylight nRMSE', lowerBetter: true },
] as const

type MetricKey = (typeof METRICS)[number]['key']

/** Model Comparison (PRD §38): every evaluated model against one protocol
 * (D-011 test split, ALL rows) with a selectable metric. Best model takes
 * the series blue; the rest recede to gray — emphasis, not rainbow. */
export default function Comparison() {
  const [metric, setMetric] = useState<MetricKey>('mae')
  const spec = METRICS.find((m) => m.key === metric)!

  const persistence = useApi(() => api.metrics('persistence'), [])
  const xgboost = useApi(() => api.metrics('xgboost'), [])
  const lstm = useApi(() => api.metrics('lstm'), [])
  const gru = useApi(() => api.metrics('gru'), [])
  const transformer = useApi(() => api.metrics('transformer'), [])

  const all: Metrics[] = useMemo(
    () =>
      [
        persistence.data,
        xgboost.data,
        lstm.data,
        gru.data,
        transformer.data,
      ].filter(Boolean) as Metrics[],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [persistence.data, xgboost.data, lstm.data, gru.data, transformer.data],
  )

  const ordered = useMemo(
    () =>
      MODEL_ORDER.map((id) => all.find((m) => m.model_id === id))
        .filter(Boolean)
        .map((m) => ({ ...m!, value: m![metric] as number | null })),
    [all, metric],
  )
  const loading = ordered.some((o) => o.value === undefined)

  const chartData = useMemo(() => {
    const vals = ordered
      .filter((o) => o.value !== null)
      .sort((a, b) =>
        spec.lowerBetter ? a.value! - b.value! : b.value! - a.value!,
      )
    return vals.map((o, i) => ({
      ...o,
      rank: i,
      best: i === 0 && vals.length > 1,
    }))
  }, [ordered, spec])

  function download() {
    if (!all.length) return
    downloadCsv(
      'unisolar_model_comparison_test_all.csv',
      all.map((m) => ({
        model_id: m.model_id,
        split: m.split,
        scope: m.scope,
        mae_kwh: m.mae,
        rmse_kwh: m.rmse,
        r2: m.r2,
        nrmse: m.nrmse ?? '',
        daylight_mae_kwh: m.daylight_mae ?? '',
        daylight_nrmse: m.daylight_nrmse ?? '',
        n_eval: m.n_eval,
      })),
    )
  }

  const table = (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead className="text-right">MAE</TableHead>
            <TableHead className="text-right">RMSE</TableHead>
            <TableHead className="text-right">R²</TableHead>
            <TableHead className="text-right">nRMSE</TableHead>
            <TableHead className="text-right">Daylight MAE</TableHead>
            <TableHead className="text-right">n eval</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {[...all]
            .sort((a, b) => a.mae - b.mae)
            .map((m) => (
              <TableRow key={m.model_id}>
                <TableCell className="font-mono text-xs">{m.model_id}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtNum(m.mae)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtNum(m.rmse)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtNum(m.r2, 3)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtNum(m.nrmse, 4)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtNum(m.daylight_mae)}</TableCell>
                <TableCell className="text-right tabular-nums">{m.n_eval.toLocaleString()}</TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </div>
  )

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        One protocol, verbatim numbers: D-011 chronological test split, ALL
        rows, nRMSE over the train observed range (99.12 kWh). Random Forest
        was never implemented in this build — baselines and served models only.
      </p>

      <VizCard
        title={`Test-split ${spec.label} by model`}
        description={
          `${spec.lowerBetter ? 'Lower is better; best' : 'Higher is better; best'} = ` +
          `${chartData[0]?.model_id ?? '—'}. Hover any bar for the exact value.` +
          ' Table view has the full metric set.'
        }
        action={
          <>
            <Select value={metric} onValueChange={(v) => setMetric(v as MetricKey)}>
              <SelectTrigger className="h-7 w-44 text-xs" aria-label="Metric">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {METRICS.map((m) => (
                  <SelectItem key={m.key} value={m.key}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={download} disabled={!all.length}>
              <Download /> CSV
            </Button>
          </>
        }
        className={loading ? 'opacity-60 transition-opacity' : ''}
        table={table}
      >
        <figure className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 56, left: 8, bottom: 0 }}
            >
              <XAxis
                type="number"
                tick={{ ...axisTick }}
                tickLine={false}
                axisLine={{ stroke: 'var(--viz-axis)' }}
                // bar length encodes the value — axis must start at zero
                domain={[0, (dataMax: number) => dataMax * 1.08]}
                tickFormatter={(v: number) =>
                  fmtNum(v, spec.key === 'r2' ? 2 : spec.key.includes('nrmse') ? 3 : 1)
                }
              />
              <YAxis
                type="category"
                dataKey="model_id"
                width={92}
                tick={{ ...axisTick }}
                tickLine={false}
                axisLine={false}
              />
              <Bar dataKey="value" name={spec.label} barSize={18} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {chartData.map((d) => (
                  <Cell key={d.model_id} fill={d.best ? 'var(--series-forecast)' : 'var(--viz-axis)'} />
                ))}
                <LabelList
                  dataKey="value"
                  position="right"
                  formatter={(v) =>
                    fmtNum(
                      Number(v),
                      spec.key === 'r2' ? 3 : spec.key.includes('nrmse') ? 4 : 2,
                    )
                  }
                  style={{
                    fontSize: 11,
                    fontFamily: 'var(--font-mono)',
                    fill: 'var(--muted-foreground)',
                  }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </figure>
      </VizCard>

      <PredActualExplorer />
      <AllRunsBars />
      <CrossSiteCard />
    </div>
  )
}

/** All 8 evaluated runs on one selectable compact metric. */
function AllRunsBars() {
  const [metric, setMetric] = useState<CompactKey>('mae')
  const spec = COMPACT_METRICS.find((m) => m.key === metric)!
  const rows = RUN_ORDER
    .filter((r) => ev.models[r])
    .map((r) => ({ run: r, value: ev.models[r].metrics[metric] as number }))
    .filter((r) => Number.isFinite(r.value))
    .sort((a, b) => (spec.lowerBetter ? a.value - b.value : b.value - a.value))
  const best = rows[0]?.run
  return (
    <VizCard
      title={`All evaluated runs — ${spec.label}`}
      description={`${rows.length} runs · Phase 8 comparison artifact · ${spec.lowerBetter ? 'lower' : 'higher'} is better.`}
      action={
        <Select value={metric} onValueChange={(v) => setMetric(v as CompactKey)}>
          <SelectTrigger className="h-7 w-32 text-xs" aria-label="Compact metric">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {COMPACT_METRICS.map((m) => (
              <SelectItem key={m.key} value={m.key}>{m.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
    >
      <figure className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, left: 8, bottom: 0 }}>
            <XAxis type="number" tick={{ ...axisTick }} tickLine={false}
                   axisLine={{ stroke: 'var(--viz-axis)' }}
                   domain={[0, (dataMax: number) => dataMax * 1.08]} />
            <YAxis type="category" dataKey="run" width={128}
                   tick={{ ...axisTick, fontSize: 10 }} tickLine={false} axisLine={false} />
            <RTooltip content={<ChartTooltip fmtValue={(v) => fmtNum(v, 3)} />} />
            <Bar dataKey="value" name={spec.label} barSize={14} radius={[0, 3, 3, 0]}
                 isAnimationActive={false}>
              {rows.map((d) => (
                <Cell key={d.run} fill={d.run === best ? 'var(--series-forecast)' : 'var(--viz-axis)'} />
              ))}
              <LabelList dataKey="value" position="right"
                         formatter={(v) => fmtNum(Number(v), 3)}
                         style={{ fontSize: 10, fontFamily: 'var(--font-mono)',
                                  fill: 'var(--muted-foreground)' }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </figure>
    </VizCard>
  )
}

/** Pred-vs-actual overlay + residual histogram for runs with stored test preds. */
function PredActualExplorer() {
  const [run, setRun] = useState('xgboost')
  const [site, setSite] = useState('ALL')
  const entry = ev.models[run]
  const perSiteKeys = Object.keys(entry?.daily_by_site ?? {})

  const overlayRows = useMemo(() => {
    if (!entry) return []
    if (site === 'ALL' || !entry.daily_by_site?.[site]) {
      const h = entry.hourly_all!
      // hourly index is the shared bundle-level hourly_t (identical across runs)
      return ev.hourly_t.map((t, i) => ({
        t: t.slice(5, 16),
        actual: h.actual[i],
        predicted: h.predicted[i],
      }))
    }
    const d = entry.daily_by_site[site]
    return d.actual.map((_, i) => ({
      t: String(i), // day index within test window
      actual: d.actual[i],
      predicted: d.predicted[i],
    }))
  }, [entry, site])

  const histRows = useMemo(() => {
    const h = entry?.residual_hist
    if (!h) return []
    return h.edges.slice(0, -1).map((e, i) => ({
      bin: e.toFixed(1),
      count: h.counts[i],
    }))
  }, [entry])

  return (
    <VizCard
      title="Predicted vs actual — held-out test window"
      description={
        `${ev.test_window.start.slice(0, 10)} → ${ev.test_window.end.slice(0, 10)}. ` +
        'ALL scope shows hourly means; a single site shows daily means. Nulls are honest gaps (night), never bridged.'
      }
      action={
        <div className="flex items-center gap-2">
          <Select value={run} onValueChange={setRun}>
            <SelectTrigger className="h-7 w-32 text-xs" aria-label="Run">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SERIES_RUNS.filter((r) => ev.models[r]?.hourly_all).map((r) => (
                <SelectItem key={r} value={r}>{r}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={site} onValueChange={setSite}>
            <SelectTrigger className="h-7 w-28 text-xs" aria-label="Site scope">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ALL">ALL</SelectItem>
              {perSiteKeys.map((s) => (
                <SelectItem key={s} value={s}>site {s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      }
    >
      <figure className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={overlayRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="t" tick={{ ...axisTick }} tickLine={false}
                   axisLine={{ stroke: 'var(--viz-axis)' }} minTickGap={40} />
            <YAxis width={44} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
            <RTooltip content={<ChartTooltip fmtValue={(v) => `${fmtNum(v, 2)} kW`} />} />
            <Legend wrapperStyle={legendStyle()} />
            <Line type="monotone" dataKey="actual" name="Observed"
                  stroke="var(--series-observed)" strokeWidth={2} dot={false}
                  connectNulls={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="predicted" name={`Predicted (${run})`}
                  stroke="var(--series-forecast)" strokeWidth={2} dot={false}
                  connectNulls={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </figure>
      <figure className="mt-2 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="bin" tick={{ ...axisTick, fontSize: 9 }} tickLine={false}
                   axisLine={{ stroke: 'var(--viz-axis)' }} interval={4} />
            <YAxis width={40} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
            <RTooltip content={<ChartTooltip fmtLabel={(l) => `${l} kW residual`}
                                              fmtValue={(v) => v.toLocaleString()} />} />
            <Bar dataKey="count" name="residuals" fill="var(--chart-3)"
                 radius={[2, 2, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </figure>
      <p className="mt-1 text-[11px] text-muted-foreground">
        Residual = predicted − actual, daylight-observed rows, ±10 kW bins.
      </p>
    </VizCard>
  )
}

/** Seen-val vs unseen-test MAE + unseen R² strip (Phase 9). */
function CrossSiteCard() {
  const entries = RUN_ORDER
    .filter((r) => cs.models[r]?.seen_val_all && cs.models[r]?.unseen_test_all)
    .map((r) => ({
      model: r,
      seen: cs.models[r].seen_val_all!.mae,
      unseen: cs.models[r].unseen_test_all!.mae,
    }))
  const r2Strip = Object.entries(cs.models).flatMap(([model, e]) =>
    e.unseen_site_r2.map((s) => ({ model, ...s })))
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <VizCard
        title="Cross-site generalization — MAE"
        description="Seen protocol = chronological val on training sites. Unseen = fully held-out sites' test windows (Phase 9, D-016)."
      >
        <figure className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={entries} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="model" tick={{ ...axisTick, fontSize: 9 }} tickLine={false}
                     axisLine={{ stroke: 'var(--viz-axis)' }} interval={0} angle={-30}
                     textAnchor="end" height={58} />
              <YAxis width={44} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
              <RTooltip content={<ChartTooltip fmtValue={(v) => `${fmtNum(v, 2)} kW`} />} />
              <Legend wrapperStyle={legendStyle()} />
              <Bar dataKey="seen" name="seen (val)" fill="var(--chart-2)"
                   radius={[2, 2, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="unseen" name="unseen (test)" fill="var(--chart-1)"
                   radius={[2, 2, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </figure>
      </VizCard>

      <VizCard
        title="Unseen-site R² per model"
        description="Each dot is one held-out site. Negative R² (red) = worse than predicting the mean — plant-size mix drives the extremes (compare within protocol)."
      >
        <ul className="max-h-64 space-y-1 overflow-auto text-xs">
          {r2Strip.sort((a, b) => a.r2 - b.r2).map((s) => (
            <li key={`${s.model}-${s.site_id}`} className="flex items-center gap-2">
              <span className="w-28 shrink-0 truncate font-mono">{s.model}</span>
              <span className="w-14 shrink-0 text-right font-mono text-muted-foreground">
                site {s.site_id}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                <div
                  className={'h-full rounded-full ' + (s.r2 < 0 ? 'bg-status-bad-text' : 'bg-chart-2')}
                  style={{ width: `${Math.min(100, Math.max(2, Math.abs(s.r2) * 100))}%` }}
                />
              </div>
              <span className={'w-14 shrink-0 text-right font-mono tabular-nums ' +
                               (s.r2 < 0 ? 'text-status-bad-text' : '')}>
                {fmtNum(s.r2, 3)}
              </span>
            </li>
          ))}
        </ul>
      </VizCard>
    </div>
  )
}
