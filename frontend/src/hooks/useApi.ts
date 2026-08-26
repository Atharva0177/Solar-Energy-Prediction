import { useEffect, useRef, useState } from 'react'

export interface ApiState<T> {
  data: T | undefined
  error: string | null
  loading: boolean
}

/** Fetch helper that HOLDS the previous render at reduced opacity while
 * refetching (no skeleton flash), and cancels stale responses. */
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[],
): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({
    data: undefined,
    error: null,
    loading: true,
  })
  const seq = useRef(0)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  useEffect(() => {
    const id = ++seq.current
    setState((prev) => ({ ...prev, loading: true, error: null }))
    fetcherRef.current().then(
      (data) => {
        if (seq.current === id) setState({ data, error: null, loading: false })
      },
      (err: Error) => {
        if (seq.current === id)
          setState((prev) => ({
            data: prev.data,
            error: err.message,
            loading: false,
          }))
      },
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}
