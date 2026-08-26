import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Download } from 'lucide-react'

import { useSites } from '@/components/site-context'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import {
  ChartTooltip,
  VizCard,
  axisTick,
  gridProps,
  legendStyle,
} from '@/components/viz'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { downloadCsv, fmtNum } from '@/lib/format'
import type { ForecastResponse, PredictionPoint } from '@/lib/types'

/** Forecasts are deterministic per (site, model, horizon) — the recursion
 * always starts at the dataset's last observation (D-019) — so responses
 * are cached for instant re-selection of a model/horizon combination.
 * Promises are cached (not results), so concurrent selects share one call. */
const fcCache = new Map<string, Promise<ForecastResponse>>()

/** Forecast (PRD §38): interactive actual-vs-forecast with site / date /
 * horizon / model controls and downloadable predictions (PRD §39).
 *
 * The served models recurse from the dataset's last observation (D-019) —
 * the date range picks which observed history is plotted alongside, it
 * cannot move the forecast origin. That constraint is stated on the card. */
export default function Forecast() {
  const { sites, selected } = useSites()
  const siteId = selected ?? 0

  const models = useApi(() => api.models(), [])

  const [model, setModel] = useState('xgboost')
  // only xgboost carries Phase 11 conformal calibration (D-022)
  const hasBounds = model === 'xgboost'
  // slider moves update the label instantly; the request follows debounced
  // so a drag doesn't fire one forecast call per step (D-019 recursion
  // costs real server work even at ~0.2 s per call)
  const [horizon, setHorizon] = useState(96)
  const [reqHorizon, setReqHorizon] = useState(96)

  useEffect(() => {
    const id = setTimeout(() => setReqHorizon(horizon), 300)
    return () => clearTimeout(id)
  }, [horizon])
  // context window defaults get patched once the site's data extent loads
  const [range, setRange] = useState<{ start: string; end: string } | null>(null)

  const tail = useApi(
    () =>
      api
        .history(siteId, { resolution: '15min' })
        .then((h) => h.rows[h.rows.length - 1]?.timestamp.slice(0, 10) ?? ''),
    [siteId],
  )
  const dataEnd = tail.data ?? ''
  const effective =
    range ??
    (dataEnd
      ? { start: shiftIso(dataEnd, -1), end: dataEnd }
      : null)

  const history = useApi(
    () =>
      effective && dataEnd
        ? api.history(siteId, { ...effective, resolution: '15min' })
        : Promise.resolve({ rows: [] as never[], n_rows: 0, resolution: '', site_id: 0 }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [siteId, effective?.start, effective?.end, dataEnd],
  )
  const fc = useApi(
    () => {
      const key = `${siteId}|${model}|${reqHorizon}`
      let p = fcCache.get(key)
      if (!p) {
        p = api.forecast({
          site_id: siteId,
          forecast_horizon: reqHorizon,
          model,
        })
        fcCache.set(key, p)
        p.catch(() => fcCache.delete(key)) // failed calls refetch
      }
      return p
    },
    [siteId, reqHorizon, model],
  )

  const { strip, regimes } = useMemo(() => {
    const regMap = new Map<string, string>()
    const pts: Record<string, unknown>[] = (history.data?.rows ?? []).map((r) => ({
      t: r.timestamp,
      actual: r.power,
    }))
    const preds: PredictionPoint[] = fc.data?.predictions ?? []
    for (const p of preds) {
      if (p.regime) regMap.set(p.timestamp, p.regime)
      pts.push({
        t: p.timestamp,
        forecast: p.prediction,
        band:
          p.lower_bound !== undefined && p.upper_bound !== undefined
            ? // night_nolag radii dip below zero — clip display at 0
              ([Math.max(0, p.lower_bound), p.upper_bound] as [number, number])
            : undefined,
      })
    }
    return { strip: pts, regimes: regMap }
  }, [history.data, fc.data])

  const loading = history.loading || fc.loading

  function download() {
    if (!fc.data) return
    downloadCsv(
      `unisolar_forecast_site${siteId}_${model}_h${horizon}.csv`,
      fc.data.predictions.map((p) => ({
        site_id: siteId,
        model,
        timestamp: p.timestamp,
        prediction_kwh: p.prediction,
        lower_bound_90: p.lower_bound ?? '',
        upper_bound_90: p.upper_bound ?? '',
        confidence_level: p.confidence_level ?? '',
        regime: p.regime ?? '',
      })),
    )
  }

  return (
    <div className="space-y-4">
      {/* one filter row scopes everything below it */}
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3 rounded-lg border bg-card px-4 py-3">
        <div className="space-y-1.5">
          <Label className="text-xs">Model</Label>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger className="w-52 font-mono text-xs" aria-label="Model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(models.data ?? []).map((m) => (
                <SelectItem key={m.model_id} value={m.model_id} disabled={!m.served}>
                  {m.model_id}
                  {!m.served && ' — not served'}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs">Observed context</Label>
          <div className="flex items-center gap-2">
            <input
              type="date"
              className="h-8 rounded-md border bg-background px-2 font-mono text-xs"
              aria-label="Context start date"
              max={dataEnd}
              min={sites ? '2020-01-01' : undefined}
              value={effective?.start ?? ''}
              onChange={(e) =>
                setRange({ start: e.target.value, end: effective?.end ?? dataEnd })
              }
            />
            <span className="text-muted-foreground">→</span>
            <input
              type="date"
              className="h-8 rounded-md border bg-background px-2 font-mono text-xs"
              aria-label="Context end date"
              max={dataEnd}
              value={effective?.end ?? ''}
              onChange={(e) =>
                setRange({ start: effective?.start ?? shiftIso(e.target.value, -1), end: e.target.value })
              }
            />
          </div>
        </div>

        <div className="min-w-56 flex-1 space-y-1.5">
          <Label className="text-xs">
            Horizon:{' '}
            <span className="font-mono">{horizon} steps · {(horizon / 4).toFixed(2).replace(/\.?0+$/, '')} h</span>
          </Label>
          <Slider
            min={1}
            max={96}
            step={1}
            value={[horizon]}
            onValueChange={([v]) => setHorizon(v)}
            aria-label="Forecast horizon in 15-minute steps"
          />
        </div>

        <Button variant="outline" size="sm" onClick={download} disabled={!fc.data}>
          <Download /> Predictions CSV
        </Button>
      </div>

      {tail.data && (
        <p className="text-xs text-muted-foreground">
          Site {siteId} · data through{' '}
          <span className="font-mono">{tail.data}</span>. Forecasts always
          continue from that last observation (recursive serving, D-019); the
          date range chooses how much observed history plots beside them.
        </p>
      )}

      <VizCard
        title="Actual vs forecast"
        description={
          `${history.data?.rows.length ?? 0} observed slots, then ${horizon} × ` +
          `15-min ${model} steps` +
          (hasBounds
            ? ' with 90% conformal bounds (clipped at zero for display).'
            : '. No interval — conformal calibration covers xgboost only.')
        }
        className={loading ? 'opacity-60 transition-opacity' : ''}
      >
        <figure className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={strip} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="t"
                // midnight tick carries the date so consecutive days don't repeat "02:00"
                tickFormatter={(t: string) =>
                  t.slice(11, 16) === '00:00' ? t.slice(5, 10) : t.slice(11, 16)
                }
                tick={{ ...axisTick }}
                interval={Math.max(1, Math.floor(strip.length / 14))}
                minTickGap={24}
                tickLine={false}
                axisLine={{ stroke: 'var(--viz-axis)' }}
              />
              <YAxis
                width={44}
                tick={{ ...axisTick }}
                tickLine={false}
                axisLine={false}
                label={{
                  value: 'kWh / 15 min',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fontSize: 11, fill: 'var(--viz-muted)' },
                }}
              />
              <RTooltip
                content={
                  <ChartTooltip
                    fmtLabel={(l: string) => {
                      const base = `${l.slice(0, 10)}, ${l.slice(11, 16)}`
                      return regimes.has(l)
                        ? `${base} · ${(regimes.get(l) as string).replace('_', '·')}`
                        : base
                    }}
                    fmtValue={(v: number) => `${fmtNum(v)} kWh`}
                  />
                }
              />
              <Legend wrapperStyle={legendStyle()} iconType="plainline" />
              {hasBounds && (
                <Area
                  dataKey="band"
                  name="90% bounds"
                  fill="var(--series-forecast)"
                  fillOpacity={0.14}
                  stroke="none"
                  isAnimationActive={false}
                  legendType="rect"
                />
              )}
              <ReferenceLine y={0} stroke="var(--viz-axis)" />
              <Line
                type="monotone"
                dataKey="actual"
                name="Observed"
                stroke="var(--series-observed)"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="forecast"
                name={`Forecast (${model})`}
                stroke="var(--series-forecast)"
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </figure>
      </VizCard>
    </div>
  )
}

function shiftIso(iso: string, days: number): string {
  const d = new Date(iso + 'T00:00:00Z')
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}
