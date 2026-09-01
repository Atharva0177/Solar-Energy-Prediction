/** Minimal typed client for the FastAPI service. Dev traffic goes through
 * the Vite proxy (/api → 127.0.0.1:8000); production wiring is Phase 14. */

import type {
  DatasetInfo,
  ForecastResponse,
  HistoryResponse,
  JobStatusResponse,
  Metrics,
  ModelEntry,
  Site,
  StartJobResponse,
  TrainConfig,
  VerifiedDataset,
} from './types'

const BASE = '/api/v1'

class ApiError extends Error {
  status: number
  /** Raw FastAPI `detail` when it is structured (e.g. train verify's
   * {message, files}); undefined for plain-string details. */
  detail?: unknown
  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

function _errorFrom(res: Response, body: unknown): ApiError {
  let message = res.statusText
  // HTTPException(detail={...}) nests under .detail; validation errors are arrays
  const d = (body as { detail?: unknown })?.detail ?? body
  if (typeof d === 'string') {
    message = d
  } else if (d && typeof d === 'object' && 'message' in (d as object)) {
    message = String((d as { message: unknown }).message)
  }
  return new ApiError(res.status, message, d)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    try {
      throw _errorFrom(res, await res.json())
    } catch (e) {
      if (e instanceof ApiError) throw e
      throw new ApiError(res.status, res.statusText)
    }
  }
  return res.json() as Promise<T>
}

/** Multipart variant — no JSON content-type; the browser sets the boundary. */
async function upload<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(BASE + path, { method: 'POST', body: form })
  if (!res.ok) {
    try {
      throw _errorFrom(res, await res.json())
    } catch (e) {
      if (e instanceof ApiError) throw e
      throw new ApiError(res.status, res.statusText)
    }
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; n_sites: number }>('/health'),
  dataset: () => request<DatasetInfo>('/dataset'),
  sites: () => request<{ sites: Site[] }>('/sites').then((r) => r.sites),
  models: () =>
    request<{ models: ModelEntry[] }>('/models').then((r) => r.models),
  metrics: (modelId: string) =>
    request<Metrics>(`/models/${modelId}/metrics`),
  history: (
    siteId: number,
    params: { start?: string; end?: string; resolution?: string } = {},
  ) => {
    const q = new URLSearchParams()
    if (params.start) q.set('start', params.start)
    if (params.end) q.set('end', params.end)
    if (params.resolution) q.set('resolution', params.resolution)
    const qs = q.toString()
    return request<HistoryResponse>(
      `/sites/${siteId}/history${qs ? `?${qs}` : ''}`,
    )
  },
  forecast: (body: {
    site_id: number
    forecast_horizon: number
    model: string
  }) =>
    request<ForecastResponse>('/forecast', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /* ---- Train page (job API, D-024/D-025) ---- */
  verifyPathDataset: (path: string) =>
    request<VerifiedDataset>('/train/datasets/path', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  listDatasets: () =>
    request<{ datasets: Array<{ path: string; description: string }> }>('/train/datasets/list'),
  uploadDataset: (files: File[]) => {
    const form = new FormData()
    for (const f of files) form.append('files', f, f.name)
    return upload<VerifiedDataset>('/train/datasets/upload', form)
  },
  startJob: (body: { dataset_id: string; model: string; fast_test?: boolean }) =>
    request<StartJobResponse>('/train/jobs', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  jobStatus: (jobId: string) =>
    request<JobStatusResponse>(`/train/jobs/${encodeURIComponent(jobId)}`),
  trainConfig: () => request<TrainConfig>('/train/config'),
}

export { ApiError }
