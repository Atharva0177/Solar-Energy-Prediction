import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
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
import { VizCard, axisTick } from '@/components/viz'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { downloadCsv, fmtNum } from '@/lib/format'
import type { Metrics } from '@/lib/types'

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
    </div>
  )
}
