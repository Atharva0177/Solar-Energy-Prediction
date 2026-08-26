import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useSites } from '@/components/site-context'
import {
  ChartTooltip,
  StatTile,
  VizCard,
  axisTick,
  gridProps,
  legendStyle,
} from '@/components/viz'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { addDays, fmtNum } from '@/lib/format'
import type { Metrics, SiteMonthlyBundle } from '@/lib/types'
import siteMonthlyJson from '@/data/site_monthly.json'

interface StripPoint {
  t: string
  actual?: number | null
  forecast?: number | null
  band?: [number, number]
  isDaylight?: boolean
}

const monthly = siteMonthlyJson as SiteMonthlyBundle

/** Monthly-energy rows for one site — null months become 0 for the bars. */
function siteMonthlyRows(siteId: number) {
  const row = monthly.sites.find((s) => s.site_id === siteId)
  if (!row) return []
  return monthly.months.map((m, i) => ({
    month: m.slice(2), // "2020-01" → "20-01" keeps ticks compact
    kwh: row.monthly_kwh[i] ?? 0,
  }))
}

/** Campus-comparison rows: mean monthly kWh per campus across shared months. */
function campusRows() {
  return monthly.campuses.map((c) => ({
    campus: `campus ${c.campus_id}`,
    nSites: c.n_sites,
    kwhMean: c.monthly_kwh_mean,
  }))
}

/** Dashboard (PRD §38): latest power, last-day energy, predicted next 24 h,
 * best model + accuracy — over one continuous day-strip: the observed day
 * flowing into tomorrow's recursive forecast with conformal bounds. */
export default function Dashboard() {
  const { selected } = useSites()
  const siteId = selected ?? 0

  const history = useApi(
    () => api.history(siteId, { resolution: '15min' }),
    [siteId],
  )
  const forecast = useApi(
    () => api.forecast({ site_id: siteId, forecast_horizon: 96, model: 'xgboost' }),
    [siteId],
  )
  const models = useApi(() => api.models(), [])
  const persistence = useApi(() => api.metrics('persistence'), [])
  const xgboost = useApi(() => api.metrics('xgboost'), [])

  const { strip, nightSpans, lastDay } = useMemo(() => {
    const rows = history.data?.rows ?? []
    if (!rows.length)
      return { strip: [] as StripPoint[], nightSpans: [], lastDay: '' }
    const today = rows[rows.length - 1].timestamp.slice(0, 10)
    const recent = rows.filter((r) => r.timestamp.slice(0, 10) >= addDays(today, -1))

    const strip: StripPoint[] = [
      ...recent.map((r) => ({
        t: r.timestamp,
        actual: r.power,
        isDaylight: r.is_daylight === true,
      })),
      ...(forecast.data?.predictions ?? []).map((p) => ({
        t: p.timestamp,
        forecast: p.prediction,
        band:
          p.lower_bound !== undefined && p.upper_bound !== undefined
            ? // night_nolag radii are wide and dip below zero — clip display at 0
              ([Math.max(0, p.lower_bound), p.upper_bound] as [number, number])
            : undefined,
      })),
    ]

    const spans: { from: string; to: string }[] = []
    let open: string | null = null
    for (const p of strip) {
      if (p.isDaylight === undefined) continue // forecast tail: no flag
      if (!p.isDaylight && open === null) open = p.t
      if (p.isDaylight && open !== null) {
        spans.push({ from: open, to: p.t })
        open = null
      }
    }
    return { strip, nightSpans: spans, lastDay: today }
  }, [history.data, forecast.data])

  const latest = useMemo(() => {
    const rows = history.data?.rows ?? []
    for (let i = rows.length - 1; i >= 0; i--)
      if (rows[i].power !== null) return rows[i].power
    return null
  }, [history.data])

  const dayEnergy = useMemo(() => {
    const rows = history.data?.rows ?? []
    const sum = rows
      .filter((r) => r.timestamp.startsWith(lastDay))
      .reduce((acc, r) => acc + (r.power ?? 0), 0)
    return sum || null
  }, [history.data, lastDay])

  const best =
    [persistence.data, xgboost.data]
      .filter((m): m is Metrics => Boolean(m))
      .reduce<Metrics | null>((a, b) => (a === null || b.mae <= a.mae ? b : a), null) ??
    undefined

  const fcSum = (forecast.data?.predictions ?? []).reduce(
    (acc, p) => acc + (p.prediction ?? 0),
    0,
  )
  const loading = history.loading || forecast.loading

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          label={`Latest power · site ${siteId}`}
          value={latest !== null ? `${fmtNum(latest)} kWh` : '—'}
          loading={history.loading}
          sub={<span className="font-mono">15-min slot, through {lastDay}</span>}
        />
        <StatTile
          label={`Energy, ${lastDay || '—'}`}
          value={dayEnergy ? `${fmtNum(dayEnergy, 1)} kWh` : '—'}
          loading={history.loading}
          sub="sum of observed slots"
        />
        <StatTile
          label="Predicted next 24 h"
          value={forecast.data ? `${fmtNum(fcSum, 1)} kWh` : '—'}
          loading={forecast.loading}
          sub="xgboost recursive · ∑ 96 × ¼ h"
        />
        <StatTile
          label="Best model (test ALL)"
          value={best?.model_id ?? '…'}
          loading={!best}
          sub={
            best && (
              <span className="font-mono">
                MAE {fmtNum(best.mae)} · R² {fmtNum(best.r2, 3)}
              </span>
            )
          }
        />
      </div>

      <VizCard
        title={`Site ${siteId} — yesterday into the next 24 hours`}
        description={
          `Observed power through ${lastDay}, then the recursive xgboost ` +
          'forecast with 90% conformal bounds (clipped at zero for display). ' +
          'Shading marks night.'
        }
        action={
          <Link
            to="/forecast"
            className="text-xs text-primary underline-offset-4 hover:underline"
          >
            Forecast controls →
          </Link>
        }
        className={loading ? 'opacity-60 transition-opacity' : ''}
      >
        <figure className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={strip} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              {nightSpans.map((s, i) => (
                <ReferenceArea
                  key={i}
                  x1={s.from}
                  x2={s.to}
                  // span bounds are exact axis categories; 'extendDomain' guards
                  // against the default 'discard' silently dropping the rect
                  ifOverflow="extendDomain"
                  fill="var(--viz-muted)"
                  fillOpacity={0.18}
                  stroke="none"
                />
              ))}
              <XAxis
                dataKey="t"
                // midnight tick carries the date so consecutive days don't repeat "02:00"
                tickFormatter={(t: string) =>
                  t.slice(11, 16) === '00:00' ? t.slice(5, 10) : t.slice(11, 16)
                }
                tick={{ ...axisTick }}
                interval={Math.max(1, Math.floor(strip.length / 12))}
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
                    fmtLabel={(l: string) =>
                      `${l.slice(0, 10)}, ${l.slice(11, 16)}`
                    }
                    fmtValue={(v: number) => `${fmtNum(v)} kWh`}
                  />
                }
              />
              <Legend wrapperStyle={legendStyle()} iconType="plainline" />
              <Area
                dataKey="band"
                name="90% bounds"
                fill="var(--series-forecast)"
                fillOpacity={0.14}
                stroke="none"
                isAnimationActive={false}
                legendType="rect"
              />
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
                name="Forecast (xgboost)"
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

      <div className="grid gap-4 lg:grid-cols-2">
        <VizCard
          title={`Site ${siteId} — monthly energy`}
          description="Sum of observed 15-min slots per month (kWh). Null months = no reported data that month."
        >
          {(() => {
            const rows = siteMonthlyRows(siteId)
            if (!rows.length)
              return (
                <p className="py-16 text-center text-sm text-muted-foreground">
                  No bundled monthly data for site {siteId}.
                </p>
              )
            return (
              <figure className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="month" tick={{ ...axisTick }} tickLine={false}
                           axisLine={{ stroke: 'var(--viz-axis)' }} minTickGap={20} />
                    <YAxis width={52} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
                    <RTooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null
                        return (
                          <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
                            <div className="mb-1 font-mono text-muted-foreground">20{label}</div>
                            <div className="flex items-center gap-2 leading-5">
                              <span className="inline-block size-2 rounded-full"
                                    style={{ background: 'var(--chart-1)' }} />
                              <span className="text-muted-foreground">energy</span>
                              <span className="ml-auto pl-4 font-mono tabular-nums">
                                {fmtNum(payload[0].value as number, 0)} kWh
                              </span>
                            </div>
                          </div>
                        )
                      }}
                    />
                    <Bar dataKey="kwh" name="monthly energy" fill="var(--chart-1)"
                         radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </figure>
            )
          })()}
        </VizCard>

        <VizCard
          title="Campus comparison"
          description="Mean monthly energy per site within each campus — capacity mix, not efficiency."
        >
          <CampusChart />
        </VizCard>
      </div>

      {models.data && (
        <p className="text-xs text-muted-foreground">
          Serving {models.data.filter((m) => m.served).length} of{' '}
          {models.data.length} registered models —{' '}
          <Link to="/models" className="text-primary hover:underline">
            Model Comparison
          </Link>{' '}
          has the full test-split picture.
        </p>
      )}
    </div>
  )
}

const CAMPUS_PALETTE = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)']

/** Mean monthly kWh per site, one line per campus (site_monthly.json). */
function CampusChart() {
  const rows = monthly.months.map((m, i) => {
    const pt: Record<string, string | number | null> = { month: m.slice(2) }
    for (const c of monthly.campuses) pt[`c${c.campus_id}`] = c.monthly_kwh_mean[i]
    return pt
  })
  return (
    <figure className="h-full min-h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="month" tick={{ ...axisTick }} tickLine={false}
                 axisLine={{ stroke: 'var(--viz-axis)' }} minTickGap={20} />
          <YAxis width={52} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
          <RTooltip
            content={<ChartTooltip fmtLabel={(l) => `20${l}`}
                                   fmtValue={(v) => `${fmtNum(v, 0)} kWh`} />}
          />
          <Legend wrapperStyle={legendStyle()} />
          {monthly.campuses.map((c, i) => (
            <Line
              key={c.campus_id}
              type="monotone"
              dataKey={`c${c.campus_id}`}
              name={`campus ${c.campus_id} · ${c.n_sites} sites`}
              stroke={CAMPUS_PALETTE[i % CAMPUS_PALETTE.length]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </figure>
  )
}
