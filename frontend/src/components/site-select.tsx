import { useSites } from '@/components/site-context'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

/** Global site picker (PRD §39: site selection). */
export function SiteSelect() {
  const { sites, selected, setSelected } = useSites()
  return (
    <Select
      value={selected !== undefined ? String(selected) : undefined}
      onValueChange={(v) => setSelected(Number(v))}
    >
      <SelectTrigger
        className="w-40 font-mono text-xs sm:w-48"
        aria-label="Selected site"
      >
        <SelectValue placeholder="Loading sites…" />
      </SelectTrigger>
      <SelectContent className="max-h-72">
        {sites?.map((s) => (
          <SelectItem key={s.site_id} value={String(s.site_id)}>
            Site {s.site_id} · campus {s.campus_id}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
