import { Route, Routes } from 'react-router-dom'

import { AppLayout } from '@/components/layout'
import { SiteProvider } from '@/components/site-context'
import Comparison from '@/pages/comparison'
import Dashboard from '@/pages/dashboard'
import Explainability from '@/pages/explainability'
import Forecast from '@/pages/forecast'
import Quality from '@/pages/quality'
import Sites from '@/pages/sites'
import Train from '@/pages/train'

export default function App() {
  return (
    <SiteProvider>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/sites" element={<Sites />} />
          <Route path="/models" element={<Comparison />} />
          <Route path="/explain" element={<Explainability />} />
          <Route path="/quality" element={<Quality />} />
          <Route path="/train" element={<Train />} />
        </Routes>
      </AppLayout>
    </SiteProvider>
  )
}
