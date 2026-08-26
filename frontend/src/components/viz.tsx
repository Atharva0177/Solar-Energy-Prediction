import { useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { TableIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { cn } from '@/lib/utils'

/** Shared Recharts axis/grid chrome — hairlines one shade off the surface,
 * mono ticks in muted ink. */
export const axisTick = {
  fill: 'var(--viz-muted)',
  fontSize: 11,
  fontFamily: 'var(--font-mono)',
}

export const gridProps = {
  stroke: 'var(--viz-grid)',
  strokeDasharray: undefined as string | undefined,
  vertical: false,
}

export function ChartTooltip({
  active,
  payload,
  label,
  fmtValue,
  fmtLabel,
}: {
  active?: boolean
  payload?: { name?: string; value?: number | [number, number]; color?: string; dataKey?: string }[]
  label?: string
  fmtValue?: (v: number) => string
  fmtLabel?: (l: string) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
      {label !== undefined && (
        <div className="mb-1 font-mono text-muted-foreground">
          {fmtLabel ? fmtLabel(label) : label}
        </div>
      )}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 leading-5">
          <span
            className="inline-block size-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-muted-foreground">{p.name}</span>
          <span className="ml-auto pl-4 font-mono tabular-nums">
            {fmtValue && typeof p.value === 'number'
              ? fmtValue(p.value)
              : Array.isArray(p.value)
                ? `${p.value[0]?.toFixed(2)} – ${p.value[1]?.toFixed(2)}`
                : '—'}
          </span>
        </div>
      ))}
    </div>
  )
}

export function legendStyle() {
  return { fontSize: 12, color: 'var(--muted-foreground)' }
}

export { CartesianGrid, Legend, Line, ResponsiveContainer, RTooltip, XAxis, YAxis }

/** Metric stat tile — hero numbers stay in the UI sans, proportional figures. */
export function StatTile({
  label,
  value,
  sub,
  loading,
}: {
  label: string
  value: string
  sub?: React.ReactNode
  loading?: boolean
}) {
  return (
    <Card className="gap-2 py-4">
      <CardHeader className="px-4">
        <CardDescription className="text-xs">{label}</CardDescription>
        <CardTitle
          className={cn(
            'text-2xl font-semibold tracking-tight',
            loading && 'opacity-40',
          )}
        >
          {loading ? '…' : value}
        </CardTitle>
      </CardHeader>
      {sub && (
        <CardContent className="px-4 text-xs text-muted-foreground">{sub}</CardContent>
      )}
    </Card>
  )
}

/** Regime badge in mono — the conformal regime the API priced this step with. */
export function RegimeBadge({ regime }: { regime: string }) {
  const day = regime.startsWith('day')
  return (
    <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
      {day ? '☀' : '☾'} {regime.replace('_', '·')}
    </span>
  )
}

/** Chart card with a table-view twin (WCAG-clean equivalent of the plot). */
export function VizCard({
  title,
  description,
  table,
  children,
  action,
  className,
}: {
  title: string
  description?: string
  table?: React.ReactNode
  children: React.ReactNode
  action?: React.ReactNode
  className?: string
}) {
  const [showTable, setShowTable] = useState(false)
  return (
    <Card className={cn('gap-4', className)}>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
        <CardAction className="flex items-center gap-2">
          {action}
          {table && (
            <Button
              variant="outline"
              size="icon-sm"
              aria-label={showTable ? 'Show chart' : 'Show data table'}
              aria-pressed={showTable}
              onClick={() => setShowTable((v) => !v)}
            >
              <TableIcon />
            </Button>
          )}
        </CardAction>
      </CardHeader>
      <CardContent className="[&_figure]:m-0 [&_svg]:outline-none">
        {table && showTable ? table : children}
      </CardContent>
    </Card>
  )
}
