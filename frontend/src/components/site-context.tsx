import { createContext, useContext, useEffect, useState } from 'react'

import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import type { Site } from '@/lib/types'

interface SiteCtx {
  sites: Site[] | undefined
  error: string | null
  selected: number | undefined
  setSelected: (id: number) => void
}

const SiteContext = createContext<SiteCtx>({
  sites: undefined,
  error: null,
  selected: undefined,
  setSelected: () => {},
})

export const useSites = () => useContext(SiteContext)

/** Loads the site list once and shares it + the selected site app-wide. */
export function SiteProvider({ children }: { children: React.ReactNode }) {
  const { data, error } = useApi(() => api.sites(), [])
  const [picked, setPicked] = useState<number | null>(() => {
    const v = Number(localStorage.getItem('unisolar-site'))
    return Number.isFinite(v) && v > 0 ? v : null
  })

  // fall back to the stored pick if valid, else the first site
  const selected =
    data?.find((s) => s.site_id === picked)?.site_id ?? data?.[0]?.site_id

  useEffect(() => {
    if (selected !== undefined)
      localStorage.setItem('unisolar-site', String(selected))
  }, [selected])

  return (
    <SiteContext.Provider
      value={{
        sites: data,
        error,
        selected,
        setSelected: (id: number) => setPicked(id),
      }}
    >
      {children}
    </SiteContext.Provider>
  )
}
