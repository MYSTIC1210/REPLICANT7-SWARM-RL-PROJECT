import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import Replicant7 from '../swarm_rl.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Replicant7 />
  </StrictMode>
)
