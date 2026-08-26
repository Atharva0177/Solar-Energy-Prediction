/** Formatting helpers. Timestamps from the API are naive Melbourne wall
 * time (D-007) — sliced as strings so the browser timezone never shifts them. */

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return v.toFixed(digits)
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

/** "2022-04-23T14:30:00" → "23 Apr 2022" */
export function fmtDate(ts: string): string {
  const [y, m, d] = ts.slice(0, 10).split('-')
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${Number(d)} ${months[Number(m) - 1]} ${y}`
}

/** → "14:30" */
export function fmtTime(ts: string): string {
  return ts.slice(11, 16)
}

/** → "23 Apr, 14:30" */
export function fmtDateTime(ts: string): string {
  return `${fmtDate(ts)}, ${fmtTime(ts)}`
}

export function dateOf(ts: string): string {
  return ts.slice(0, 10)
}

export function addDays(isoDate: string, days: number): string {
  const d = new Date(isoDate + 'T00:00:00Z')
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

/** Client-side CSV download (PRD §39: downloadable predictions/evaluation). */
export function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return
  const cols = Object.keys(rows[0])
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s
  }
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
