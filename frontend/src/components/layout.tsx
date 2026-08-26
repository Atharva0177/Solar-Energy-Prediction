import { NavLink, useLocation } from 'react-router-dom'
import {
  Activity,
  Cpu,
  GitCompareArrows,
  LayoutDashboard,
  MapPin,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

import { SiteSelect } from '@/components/site-select'
import { ThemeToggle } from '@/components/theme'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/forecast', label: 'Forecast', icon: Activity },
  { to: '/sites', label: 'Sites', icon: MapPin },
  { to: '/models', label: 'Model Comparison', icon: GitCompareArrows },
  { to: '/explain', label: 'Explainability', icon: Sparkles },
  { to: '/quality', label: 'Data Quality', icon: ShieldCheck },
  { to: '/train', label: 'Train Models', icon: Cpu },
]

export function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-svh flex-col md:flex-row">
      <aside className="flex shrink-0 flex-col border-b bg-sidebar md:border-b-0 md:border-r md:w-56">
        <div className="flex items-center gap-2 px-4 py-4">
          <SunMark />
          <div>
            <div className="text-sm font-semibold leading-tight tracking-tight">
              UNISOLAR
            </div>
            <div className="font-mono text-[10px] text-muted-foreground">
              solar forecast console
            </div>
          </div>
        </div>
        <nav aria-label="Primary" className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors',
                  isActive
                    ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
                    : 'text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground',
                )
              }
            >
              <Icon className="size-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80 sm:px-6">
          <PageTitle />
          <div className="flex items-center gap-3">
            <SiteSelect />
            <ThemeToggle />
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
          {children}
        </main>
      </div>
    </div>
  )
}

function PageTitle() {
  const loc = useLocation()
  const current =
    NAV.find((n) => (n.to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(n.to)))
      ?.label ?? 'UNISOLAR'
  return (
    <h1 className="truncate text-sm font-medium text-muted-foreground">{current}</h1>
  )
}

/** Wordmark sun — half above the horizon line, southern-hemisphere north-facing light. */
function SunMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-6" aria-hidden="true">
      <circle cx="12" cy="13" r="5" fill="var(--primary)" />
      <rect x="1" y="13" width="22" height="1.4" fill="var(--foreground)" opacity="0.25" />
      <g stroke="var(--primary)" strokeWidth="1.6" strokeLinecap="round">
        <line x1="12" y1="3.2" x2="12" y2="5.4" />
        <line x1="5.4" y1="6.4" x2="7" y2="8" />
        <line x1="18.6" y1="6.4" x2="17" y2="8" />
      </g>
    </svg>
  )
}
