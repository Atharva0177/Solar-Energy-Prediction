import { useState } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { VizCard, axisTick } from '@/components/viz'
import { fmtNum } from '@/lib/format'
import type { ShapBundle, ShapFeature } from '@/lib/types'
import shapBundleJson from '@/data/shap_global_importance.json'

const bundle = shapBundleJson as ShapBundle
const TOP_N = 15

/** Static SHAP figures from the Phase 10 run, served read-only via /static. */
const SHAP_PANELS = [
  { file: 'shap_summary_bar.png', title: 'Summary (bar) — mean |SHAP| per feature',
    note: 'Same ranking as the interactive chart above, from the Phase 10 run.' },
  { file: 'shap_beeswarm.png', title: 'Beeswarm — value vs impact',
    note: 'Each dot is one sample; red = high feature value. power_lag_1 low → negative contribution.' },
  { file: 'shap_waterfall_clear_noon_peak.png', title: 'Waterfall — clear noon peak',
    note: 'Recent lags push the prediction to the daily maximum.' },
  { file: 'shap_waterfall_morning_ramp.png', title: 'Waterfall — morning ramp',
    note: 'Lag and rolling-mean features disagree — the ramp is where the model leans on trend.' },
  { file: 'shap_waterfall_overcast_afternoon.png', title: 'Waterfall — overcast afternoon',
    note: 'Weather features pull the forecast below the clear-sky lag anchor.' },
  { file: 'shap_waterfall_night_zero.png', title: 'Waterfall — night zero',
    note: 'The failure mode: missing recent history lets daytime-scale features fire at night.' },
  { file: 'shap_dependence_1_power_lag_1.png', title: 'Dependence — power_lag_1',
    note: 'Near-linear main driver; saturation near plant capacity.' },
  { file: 'shap_dependence_2_power_rolling_mean_3600s.png', title: 'Dependence — rolling mean 1 h',
    note: 'Smoothed context feature; wide spread at mid values marks transition hours.' },
]

/** Explainability (PRD §38): SHAP global importance for the served XGBoost
 * run (exact TreeExplainer, D-017). Features are nominal categories → one
 * series color for every bar (no magnitude ramp); the value axis carries size. */
export default function Explainability() {
  const [showAll, setShowAll] = useState(false)
  const rows = showAll ? bundle.features : bundle.features.slice(0, TOP_N)
  const topShare = bundle.features.slice(0, 2).reduce((a, f) => a + f.share, 0)

  const table = (
    <div className="max-h-96 overflow-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>#</TableHead>
            <TableHead>Feature</TableHead>
            <TableHead className="text-right">mean |SHAP| kWh</TableHead>
            <TableHead className="text-right">share</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {bundle.features.map((f: ShapFeature) => (
            <TableRow key={f.feature}>
              <TableCell className="font-mono text-xs">{f.rank}</TableCell>
              <TableCell className="font-mono text-xs">{f.feature}</TableCell>
              <TableCell className="text-right tabular-nums">{fmtNum(f.mean_abs_shap)}</TableCell>
              <TableCell className="text-right tabular-nums">{(f.share * 100).toFixed(1)}%</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Exact TreeExplainer attributions for{' '}
        <code className="font-mono text-xs">{bundle.explained_run}</code>,
        computed on a seeded {bundle.sample_rows.toLocaleString()}-row sample of
        the D-011 test split. Additivity error{' '}
        {bundle.additivity_max_abs_err.toExponential(1)}. The top two features —
        the last observed slot and its trailing-hour mean — carry{' '}
        {(topShare * 100).toFixed(0)}% of total attribution; weather is minor.
      </p>

      <VizCard
        title={`Global feature importance — ${rows.length === bundle.features.length ? 'all' : `top ${TOP_N}`} features`}
        description="Bar length = mean |SHAP| (kWh). Hover for exact values; table view lists all features."
        action={
          <button
            className="text-xs text-primary underline-offset-4 hover:underline"
            onClick={() => setShowAll((v) => !v)}
          >
            {showAll ? 'Show top only' : 'Show all'}
          </button>
        }
        table={table}
      >
        <figure className={showAll ? 'h-[32rem]' : 'h-96'}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={rows}
              layout="vertical"
              margin={{ top: 4, right: 64, left: 8, bottom: 0 }}
            >
              <XAxis
                type="number"
                tick={{ ...axisTick }}
                tickLine={false}
                axisLine={{ stroke: 'var(--viz-axis)' }}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={210}
                tick={{ ...axisTick }}
                tickLine={false}
                axisLine={false}
              />
              <RTooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null
                  const f = payload[0].payload as ShapFeature
                  return (
                    <div className="rounded-lg border bg-popover px-3 py-2 font-mono text-xs shadow-md">
                      <div className="mb-1 text-muted-foreground">#{f.rank} {f.feature}</div>
                      <div>mean |SHAP| {fmtNum(f.mean_abs_shap)} kWh</div>
                      <div>share {(f.share * 100).toFixed(1)}%</div>
                    </div>
                  )
                }}
              />
              <Bar dataKey="mean_abs_shap" name="mean |SHAP|" barSize={16} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {rows.map((f) => (
                  <Cell key={f.feature} fill="var(--series-forecast)" />
                ))}
                <LabelList
                  dataKey="share"
                  position="right"
                  formatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
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

      <p className="text-xs text-muted-foreground">
        Reading note: shares are of total absolute attribution on the test
        split. The night failure mode (missing recent history → daytime-scale
        predictions) and per-scenario waterfalls live in{' '}
        <code className="font-mono text-xs">artifacts/shap/</code>; see
        RESULTS.md for the quantified night breakdown.
      </p>

      {/* Static SHAP gallery — read-only artifact PNGs via the /static mount */}
      <section aria-label="SHAP gallery" className="space-y-2">
        <h2 className="text-sm font-medium">SHAP figures — Phase 10 run</h2>
        <p className="text-xs text-muted-foreground">
          Beeswarm, per-scenario waterfalls and dependence plots from{' '}
          <code className="font-mono">artifacts/shap/</code>. Click any figure for full size.
        </p>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {SHAP_PANELS.map((p) => (
            <figure key={p.file} className="overflow-hidden rounded-lg border bg-card">
              <a href={`/static/shap/${p.file}`} target="_blank" rel="noreferrer">
                <img
                  src={`/static/shap/${p.file}`}
                  alt={p.title}
                  loading="lazy"
                  className="block w-full bg-background object-contain"
                />
              </a>
              <figcaption className="border-t px-3 py-1.5">
                <div className="text-[11px] font-medium">{p.title}</div>
                <div className="font-mono text-[10px] text-muted-foreground">{p.note}</div>
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    </div>
  )
}
