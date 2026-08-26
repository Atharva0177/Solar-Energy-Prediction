import { createContext, useContext, useEffect, useState } from 'react'
import { Monitor, Moon, Sun } from 'lucide-react'

import { Button } from '@/components/ui/button'

type Theme = 'light' | 'dark' | 'system'

const ThemeCtx = createContext<{ theme: Theme; setTheme: (t: Theme) => void }>({
  theme: 'system',
  setTheme: () => {},
})

function apply(theme: Theme) {
  const dark =
    theme === 'dark' ||
    (theme === 'system' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', dark)
  // data-theme stamp beats the OS setting in both directions
  document.documentElement.dataset.theme = theme === 'system' ? '' : theme
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem('unisolar-theme') as Theme) || 'system',
  )

  useEffect(() => {
    localStorage.setItem('unisolar-theme', theme)
    apply(theme)
    if (theme !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => apply('system')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [theme])

  return (
    <ThemeCtx.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeCtx.Provider>
  )
}

const OPTIONS: { value: Theme; icon: typeof Sun; label: string }[] = [
  { value: 'light', icon: Sun, label: 'Light' },
  { value: 'dark', icon: Moon, label: 'Dark' },
  { value: 'system', icon: Monitor, label: 'System' },
]

export function ThemeToggle() {
  const { theme, setTheme } = useContext(ThemeCtx)
  return (
    <div className="flex items-center rounded-lg border p-0.5">
      {OPTIONS.map(({ value, icon: Icon, label }) => (
        <Button
          key={value}
          variant="ghost"
          size="icon-sm"
          aria-label={`${label} theme`}
          aria-pressed={theme === value}
          className={theme === value ? 'bg-muted text-foreground' : 'text-muted-foreground'}
          onClick={() => setTheme(value)}
        >
          <Icon />
        </Button>
      ))}
    </div>
  )
}
