import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { CheckCircle2 } from 'lucide-react'

import { StatTile, VizCard, axisTick, gridProps, legendStyle } from '@/components/viz'
import { fmtNum } from '@/lib/format'
import type { QualityBundle, QualityExtraBundle } from '@/lib/types'
import qualityJson from '@/data/data_quality.json'
import qualityExtraJson from '@/data/quality_extra.json'

const q = qualityJson as QualityBundle
const qx = qualityExtraJson as QualityExtraBundle

const EDA_PANELS = [
  { file: '01_power_distribution.png', title: 'Power distribution' },
  { file: '02_daily_profiles_campus.png', title: 'Daily profiles by campus' },
  { file: '03_monthly_energy_timeseries.png', title: 'Monthly energy' },
  { file: '04_seasonality.png', title: 'Seasonality' },
  { file: '05_site_comparison.png', title: 'Site comparison' },
  { file: '06_weather_correlation.png', title: 'Weather correlation' },
  { file: '07_missing_by_site.png', title: 'Missing by site' },
  { file: '08_missingness_heatmap.png', title: 'Missingness heatmap' },
  { file: '09_timeseries_largest_site.png', title: 'Largest-site time series' },
]

const WEATHER_SERIES = [
  { key: 'temperature_pct', name: 'Temperature', stroke: 'var(--chart-1)' },
  { key: 'humidity_pct', name: 'Humidity', stroke: 'var(--chart-2)' },
  { key: 'wind_speed_pct', name: 'Wind speed', stroke: 'var(--chart-3)' },
] as const

/** Data Quality (PRD §38): missing values, duplicates, time gaps, outliers —
 * verbatim from the Phase 2 validation artifacts. */
export default function Quality() {
  const g = q.generation
  const cleanTiles = [
    { label: 'Duplicate (site, timestamp)', value: g.duplicate_keys },
    { label: 'Impossible values', value: g.impossible_values },
    { label: 'Outliers flagged (daylight IQR)', value: q.outliers_flagged },
  ]

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        {cleanTiles.map((t) => (
          <StatTile
            key={t.label}
            label={t.label}
            value={String(t.value)}
            sub={
              t.value === 0 ? (
                <span className="inline-flex items-center gap-1 text-status-good-text">
                  <CheckCircle2 className="size-3" aria-hidden /> clean
                </span>
              ) : undefined
            }
          />
        ))}
        <StatTile
          label="Missing generation slots"
          value={g.missing_slots.toLocaleString()}
          sub={`${fmtNum((100 * g.missing_slots) / (g.groups_checked * 81000), 1)}% of expected grid · ${g.groups_checked} sites`}
        />
        <StatTile
          label="Power NaN in raw feed"
          value={`${fmtNum(g.power_missing_raw_pct, 1)}%`}
          sub="includes night reporting NaN by design"
        />
      </div>

      <VizCard
        title="Weather coverage by month"
        description="Share of campus-grain weather rows missing per month. Wind drops out entirely from Aug 2021 — models after that date run on carried-forward weather."
      >
        <figure className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChartHost />
          </ResponsiveContainer>
        </figure>
      </VizCard>

      <div className="grid gap-4 lg:grid-cols-2">
        <VizCard
          title="Weather variables — overall missing"
          description="Across the full 2020-01 → 2022-04 span."
        >
          <ul className="space-y-2 text-sm">
            {Object.entries(q.weather_overall_pct)
              .sort((a, b) => a[1] - b[1])
              .map(([k, v]) => (
                <li key={k} className="flex items-center gap-3">
                  <span className="w-44 shrink-0 font-mono text-xs">{k}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-chart-1"
                      style={{ width: `${v}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono text-xs tabular-nums">
                    {v.toFixed(1)}%
                  </span>
                </li>
              ))}
          </ul>
        </VizCard>

        <VizCard
          title="Largest per-site generation gaps"
          description="Missing 15-min slots against each site's expected grid."
        >
          <ul className="space-y-2 text-sm">
            {g.worst_sites.map((s) => (
              <li key={s.site_id} className="flex items-center gap-3">
                <span className="w-16 shrink-0 font-mono text-xs">site {s.site_id}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-chart-1"
                    style={{ width: `${(s.missing_pct / 2.5) * 100}%` }}
                  />
                </div>
                <span className="w-14 text-right font-mono text-xs tabular-nums">
                  {s.missing_pct.toFixed(2)}%
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-muted-foreground">
            Cleaning performed {q.cleaning.total_operations} logged operations;
            weather time-interpolation touched{' '}
            {q.cleaning.weather_interpolated_rows.toLocaleString()} rows
            (limit ≤ 2 steps). Power was never imputed.
          </p>
        </VizCard>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <VizCard
          title="Daylight generation distribution"
          description="Observed daylight-slot power across all sites, 0.5 kW bins (Phase 2 processed parquet). The mass near zero is overcast winter days."
        >
          <figure className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={qx.hist_bin_labels.map((label, i) => ({
                  bin: label,
                  count: qx.hist_total[i] ?? 0,
                }))}
                margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
              >
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="bin" tick={{ ...axisTick, fontSize: 9 }} tickLine={false}
                       axisLine={{ stroke: 'var(--viz-axis)' }} interval={1} />
                <YAxis width={56} tick={{ ...axisTick }} tickLine={false} axisLine={false}
                       tickFormatter={(v: number) => v.toLocaleString()} />
                <RTooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
                        <div className="mb-1 font-mono text-muted-foreground">{label} kW</div>
                        <div className="flex items-center gap-2 leading-5">
                          <span className="inline-block size-2 rounded-full"
                                style={{ background: 'var(--chart-1)' }} />
                          <span className="text-muted-foreground">slots</span>
                          <span className="ml-auto pl-4 font-mono tabular-nums">
                            {(payload[0].value as number).toLocaleString()}
                          </span>
                        </div>
                      </div>
                    )
                  }}
                />
                <Bar dataKey="count" name="daylight slots" fill="var(--chart-2)"
                     radius={[2, 2, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </figure>
        </VizCard>

        <VizCard
          title="Per-site reporting availability"
          description={
            'Row availability vs expected 15-min grid (bars), with the share of ' +
            'daylight slots carrying an observed power value (mono figure).'
          }
        >
          <ul className="space-y-2 text-sm">
            {qx.availability.map((a) => (
              <li key={a.site_id} className="flex items-center gap-3">
                <span className="w-16 shrink-0 font-mono text-xs">site {a.site_id}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className={'h-full rounded-full ' + (a.row_availability_pct >= 95 ? 'bg-chart-2' : 'bg-chart-5')}
                    style={{ width: `${a.row_availability_pct}%` }}
                  />
                </div>
                <span className="w-28 text-right font-mono text-xs tabular-nums"
                      title="row availability / daylight power observed">
                  {a.row_availability_pct.toFixed(1)}% · {a.daylight_power_obs_pct.toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        </VizCard>
      </div>

      {/* EDA gallery — read-only artifact PNGs via the /static mount */}
      <section aria-label="EDA gallery" className="space-y-2">
        <h2 className="text-sm font-medium">Exploratory analysis gallery</h2>
        <p className="text-xs text-muted-foreground">
          Static figures from the Phase 3 EDA run (<code className="font-mono">artifacts/eda/</code>,
          served read-only by the API).
        </p>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {EDA_PANELS.map((p) => (
            <figure key={p.file} className="overflow-hidden rounded-lg border bg-card">
              <a href={`/static/eda/${p.file}`} target="_blank" rel="noreferrer">
                <img
                  src={`/static/eda/${p.file}`}
                  alt={p.title}
                  loading="lazy"
                  className="block w-full bg-background object-contain"
                />
              </a>
              <figcaption className="border-t px-3 py-1.5 font-mono text-[11px] text-muted-foreground">
                {p.file}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    </div>
  )
}

function LineChartHost() {
  return (
    <LineChart data={q.weather_monthly} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
      <CartesianGrid {...gridProps} />
      <XAxis
        dataKey="month"
        tick={{ ...axisTick }}
        tickLine={false}
        axisLine={{ stroke: 'var(--viz-axis)' }}
        minTickGap={28}
      />
      <YAxis
        width={40}
        domain={[0, 100]}
        tickFormatter={(v: number) => `${v}%`}
        tick={{ ...axisTick }}
        tickLine={false}
        axisLine={false}
      />
      <RTooltip
        content={({ active, payload, label }) => {
          if (!active || !payload?.length) return null
          return (
            <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
              <div className="mb-1 font-mono text-muted-foreground">{label}</div>
              {payload.map((p, i) => (
                <div key={i} className="flex items-center gap-2 leading-5">
                  <span className="inline-block size-2 rounded-full" style={{ background: p.color }} />
                  <span className="text-muted-foreground">{p.name}</span>
                  <span className="ml-auto pl-4 font-mono tabular-nums">
                    {Number(p.value).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )
        }}
      />
      <Legend wrapperStyle={legendStyle()} />
      {WEATHER_SERIES.map((s) => (
        <Line
          key={s.key}
          type="monotone"
          dataKey={s.key}
          name={s.name}
          stroke={s.stroke}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      ))}
    </LineChart>
  )
}
