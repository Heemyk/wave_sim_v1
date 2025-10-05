import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Text } from '@react-three/drei';
import styled from 'styled-components';
import Plot from 'react-plotly.js';
import axios from 'axios';

// Declare global electronAPI for TypeScript
declare global {
  interface Window {
    electronAPI?: {
      backendRequest: (config: any) => Promise<any>;
      websocketConnect: (url: string) => Promise<any>;
      onMenuNewSimulation: (callback: () => void) => void;
      onMenuRunSimulation: (callback: () => void) => void;
      onMenuStopSimulation: (callback: () => void) => void;
      onWebSocketConnected: (callback: () => void) => void;
      onWebSocketMessage: (callback: (event: any, data: string) => void) => void;
      onWebSocketClosed: (callback: () => void) => void;
      onWebSocketError: (callback: (event: any, error: string) => void) => void;
      removeAllListeners: (channel: string) => void;
    };
  }
}

// Types
interface SimulationRequest {
  name?: string;
  room: {
    type: string;
    dimensions: number[];
    center: number[];
  };
  boundaries: {
    walls: { alpha: number };
    floor: { alpha: number };
    ceiling: { alpha: number };
  };
  sources: Array<{
    id: string;
    position: number[];
    signal: {
      type: string;
      f0: number;
      f1: number;
      amplitude: number;
    };
  }>;
  mesh: {
    element_order: number;
  };
  simulation: {
    fmin: number;
    fmax: number;
    df: number;
  };
  output: {
    sensors: Array<{
      id: string;
      position: number[];
    }>;
  };
}

interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
}

// Styled components
const Container = styled.div`
  display: flex;
  height: 100vh;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const Sidebar = styled.div`
  width: 350px;
  background: #f5f5f5;
  padding: 20px;
  overflow-y: auto;
  border-right: 1px solid #ddd;
`;

const MainArea = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const Viewer = styled.div`
  flex: 1;
  background: #fff;
`;

const Controls = styled.div`
  height: 200px;
  background: #f9f9f9;
  padding: 20px;
  border-top: 1px solid #ddd;
`;

const FormGroup = styled.div`
  margin-bottom: 15px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 5px;
  font-weight: 500;
`;

const Input = styled.input`
  width: 100%;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
`;

const Select = styled.select`
  width: 100%;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
`;

const Button = styled.button`
  background: #007bff;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  
  &:hover {
    background: #0056b3;
  }
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
  }
`;

const Status = styled.div<{ status: string }>`
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 10px;
  background: ${props => {
    switch (props.status) {
      case 'completed': return '#d4edda';
      case 'failed': return '#f8d7da';
      case 'running': return '#d1ecf1';
      default: return '#f8f9fa';
    }
  }};
  color: ${props => {
    switch (props.status) {
      case 'completed': return '#155724';
      case 'failed': return '#721c24';
      case 'running': return '#0c5460';
      default: return '#495057';
    }
  }};
`;

// 3D Scene Component
const Scene3D: React.FC<{
  config: SimulationRequest;
  sources: Array<{ id: string; position: number[] }>;
  sensors: Array<{ id: string; position: number[] }>;
  meshData?: any;
  fieldData?: any;
}> = ({ config, sources, sensors, meshData, fieldData }) => {
  return (
    <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      
      {/* Room */}
      <Box 
        args={[config.room.dimensions[0], config.room.dimensions[1], config.room.dimensions[2]]} 
        position={[config.room.center[0], config.room.center[1], config.room.center[2]]}
      >
        <meshBasicMaterial color="#e0e0e0" wireframe />
      </Box>
      
      {/* Sources */}
      {sources.map((source) => (
        <Sphere 
          key={source.id} 
          position={[source.position[0], source.position[1], source.position[2]]} 
          args={[0.1]}
        >
          <meshBasicMaterial color="#ff4444" />
        </Sphere>
      ))}
      
      {/* Sensors */}
      {sensors.map((sensor) => (
        <Sphere 
          key={sensor.id} 
          position={[sensor.position[0], sensor.position[1], sensor.position[2]]} 
          args={[0.05]}
        >
          <meshBasicMaterial color="#4444ff" />
        </Sphere>
      ))}
      
      <OrbitControls />
    </Canvas>
  );
};

// Helper function for API calls
const apiCall = async (url: string, method: string = 'GET', data?: any) => {
  const baseUrl = 'http://localhost:8000';
  const fullUrl = url.startsWith('http') ? url : `${baseUrl}${url}`;
  
  if (window.electronAPI) {
    // Use Electron's secure API
    const result = await window.electronAPI.backendRequest({
      url: fullUrl,
      method,
      data
    });
    
    if (result.success) {
      return { data: result.data };
    } else {
      throw new Error(result.error);
    }
  } else {
    // Fallback to direct axios calls (for web browser)
    return await axios({
      method,
      url: fullUrl,
      data,
      timeout: 30000
    });
  }
};

// Main App Component
const App: React.FC = () => {
  const [config, setConfig] = useState<SimulationRequest>({
    name: "Frontend Test Simulation",
    room: {
      type: 'box',
      dimensions: [4, 3, 2.5],
      center: [0, 0, 0]
    },
    boundaries: {
      walls: { alpha: 0.1 },
      floor: { alpha: 0.2 },
      ceiling: { alpha: 0.15 }
    },
    sources: [{
      id: 'source1',
      position: [2, 1.5, 1],
      signal: {
        type: 'sine',
        f0: 100,
        f1: 200,
        amplitude: 1.0
      }
    }],
    mesh: {
      element_order: 1
    },
    simulation: {
      fmin: 100,
      fmax: 200,
      df: 100
    },
    output: {
      sensors: [{
        id: 'listener_1',
        position: [1, 1, 1]
      }, {
        id: 'listener_2',
        position: [3, 2, 1.5]
      }]
    }
  });

  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [results, setResults] = useState<any>(null);
  const [meshData, setMeshData] = useState<any>(null);
  const [fieldData, setFieldData] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  // Menu event handlers
  useEffect(() => {
    if (window.electronAPI) {
      window.electronAPI.onMenuRunSimulation(() => {
        if (!isRunning) {
          runSimulation();
        }
      });
      
      window.electronAPI.onMenuStopSimulation(() => {
        if (isRunning && jobStatus?.job_id) {
          // Cancel the job
          apiCall(`/api/jobs/${jobStatus.job_id}`, 'DELETE');
          setIsRunning(false);
        }
      });
      
      window.electronAPI.onMenuNewSimulation(() => {
        setResults(null);
        setMeshData(null);
        setFieldData(null);
        setJobStatus(null);
        setIsRunning(false);
      });
    }
  }, [isRunning, jobStatus?.job_id]);

  // WebSocket connection
  useEffect(() => {
    if (jobStatus?.job_id) {
      const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobStatus.job_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setJobStatus(data);
        
        if (data.status === 'completed') {
          setIsRunning(false);
          fetchResults(data.job_id);
          fetchMeshData(data.job_id);
          fetchFieldData(data.job_id, config.simulation.fmin);
        } else if (data.status === 'failed') {
          setIsRunning(false);
        }
      };

      return () => {
        ws.close();
      };
    }
  }, [jobStatus?.job_id]);

  const fetchResults = async (jobId: string) => {
    try {
      const response = await apiCall(`/api/jobs/${jobId}/results`);
      setResults(response.data);
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  const fetchMeshData = async (jobId: string) => {
    try {
      const response = await apiCall(`/api/jobs/${jobId}/mesh`);
      setMeshData(response.data);
    } catch (error) {
      console.error('Error fetching mesh data:', error);
    }
  };

  const fetchFieldData = async (jobId: string, frequency: number) => {
    try {
      const response = await apiCall(`/api/jobs/${jobId}/field/${frequency}`);
      setFieldData(response.data);
    } catch (error) {
      console.error('Error fetching field data:', error);
    }
  };

  const runSimulation = async () => {
    try {
      setIsRunning(true);
      setJobStatus(null);
      setResults(null);
      setMeshData(null);
      setFieldData(null);

      // Step 1: Submit job
      const submitResponse = await apiCall('/api/simulate', 'POST', config);
      const jobData = submitResponse.data;
      setJobStatus(jobData);

      // Step 2: Start processing
      await apiCall(`/api/simulate/start/${jobData.job_id}`, 'POST');
      
    } catch (error) {
      console.error('Error running simulation:', error);
      setIsRunning(false);
    }
  };

  const updateConfig = (path: string, value: any) => {
    setConfig(prev => {
      const newConfig = { ...prev };
      const keys = path.split('.');
      let current = newConfig;
      
      for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
      }
      
      current[keys[keys.length - 1]] = value;
      return newConfig;
    });
  };

  return (
    <Container>
      <Sidebar>
        <h2>Acoustic Simulator</h2>
        
        <FormGroup>
          <Label>Room Type</Label>
          <Select 
            value={config.room.type} 
            onChange={(e) => updateConfig('room.type', e.target.value)}
          >
            <option value="box">Box</option>
            <option value="cylinder">Cylinder</option>
            <option value="custom">Custom</option>
          </Select>
        </FormGroup>

        <FormGroup>
          <Label>Dimensions (L×W×H)</Label>
          <div style={{ display: 'flex', gap: '5px' }}>
            <Input 
              type="number" 
              value={config.room.dimensions[0]} 
              onChange={(e) => updateConfig('room.dimensions.0', parseFloat(e.target.value))}
              placeholder="Length"
            />
            <Input 
              type="number" 
              value={config.room.dimensions[1]} 
              onChange={(e) => updateConfig('room.dimensions.1', parseFloat(e.target.value))}
              placeholder="Width"
            />
            <Input 
              type="number" 
              value={config.room.dimensions[2]} 
              onChange={(e) => updateConfig('room.dimensions.2', parseFloat(e.target.value))}
              placeholder="Height"
            />
          </div>
        </FormGroup>

        <FormGroup>
          <Label>Frequency Range</Label>
          <div style={{ display: 'flex', gap: '5px' }}>
            <Input 
              type="number" 
              value={config.simulation.fmin} 
              onChange={(e) => updateConfig('simulation.fmin', parseFloat(e.target.value))}
              placeholder="Min (Hz)"
            />
            <Input 
              type="number" 
              value={config.simulation.fmax} 
              onChange={(e) => updateConfig('simulation.fmax', parseFloat(e.target.value))}
              placeholder="Max (Hz)"
            />
            <Input 
              type="number" 
              value={config.simulation.df} 
              onChange={(e) => updateConfig('simulation.df', parseFloat(e.target.value))}
              placeholder="Step (Hz)"
            />
          </div>
        </FormGroup>

        <FormGroup>
          <Label>Boundary Absorption</Label>
          <div style={{ display: 'flex', gap: '5px', flexDirection: 'column' }}>
            <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
              <span style={{ minWidth: '60px' }}>Walls:</span>
              <Input 
                type="number" 
                step="0.1"
                min="0"
                max="1"
                value={config.boundaries.walls.alpha} 
                onChange={(e) => updateConfig('boundaries.walls.alpha', parseFloat(e.target.value))}
              />
            </div>
            <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
              <span style={{ minWidth: '60px' }}>Floor:</span>
              <Input 
                type="number" 
                step="0.1"
                min="0"
                max="1"
                value={config.boundaries.floor.alpha} 
                onChange={(e) => updateConfig('boundaries.floor.alpha', parseFloat(e.target.value))}
              />
            </div>
            <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
              <span style={{ minWidth: '60px' }}>Ceiling:</span>
              <Input 
                type="number" 
                step="0.1"
                min="0"
                max="1"
                value={config.boundaries.ceiling.alpha} 
                onChange={(e) => updateConfig('boundaries.ceiling.alpha', parseFloat(e.target.value))}
              />
            </div>
          </div>
        </FormGroup>

        <Button onClick={runSimulation} disabled={isRunning}>
          {isRunning ? 'Running...' : 'Run Simulation'}
        </Button>

        {jobStatus && (
          <Status status={jobStatus.status}>
            <strong>{jobStatus.status.toUpperCase()}</strong>
            <br />
            {jobStatus.message}
            {jobStatus.progress > 0 && (
              <div style={{ marginTop: '5px' }}>
                Progress: {(jobStatus.progress * 100).toFixed(1)}%
              </div>
            )}
          </Status>
        )}
      </Sidebar>

      <MainArea>
        <Viewer>
          <Scene3D 
            config={config} 
            sources={config.sources}
            sensors={config.output.sensors}
            meshData={meshData}
            fieldData={fieldData}
          />
        </Viewer>
        
        <Controls>
          {results && (
            <div>
              <h3>Simulation Results</h3>
              {meshData && (
                <div style={{ fontSize: '12px', marginBottom: '10px' }}>
                  <strong>Mesh:</strong> {meshData.mesh?.num_vertices || 0} vertices, {meshData.mesh?.num_cells || 0} cells
                </div>
              )}
              {fieldData && (
                <div style={{ fontSize: '12px', marginBottom: '10px' }}>
                  <strong>Field:</strong> {fieldData.field_data?.num_dofs || 0} DOFs at {fieldData.frequency} Hz
                </div>
              )}
              <Plot
                data={results.frequencies?.map((freq: any) => ({
                  x: Object.keys(freq.sensor_data || {}),
                  y: Object.values(freq.sensor_data || {}).map((value: any) => {
                    // Handle complex numbers
                    if (typeof value === 'string') {
                      // Parse complex string like "1.0+0.5j"
                      const complexStr = value.replace('j', 'i');
                      const match = complexStr.match(/([+-]?\d*\.?\d+)([+-]\d*\.?\d*)i?/);
                      if (match) {
                        const real = parseFloat(match[1]);
                        const imag = match[2] ? parseFloat(match[2]) : 0;
                        return Math.sqrt(real * real + imag * imag);
                      }
                    } else if (typeof value === 'object' && value.real !== undefined) {
                      // Handle complex object {real: x, imag: y}
                      return Math.sqrt(value.real * value.real + value.imag * value.imag);
                    }
                    return Math.abs(value || 0);
                  }),
                  type: 'scatter',
                  mode: 'markers',
                  name: `${freq.frequency} Hz`
                })) || []}
                layout={{
                  title: 'Sensor Pressure Magnitudes',
                  xaxis: { title: 'Sensor ID' },
                  yaxis: { title: 'Pressure Magnitude (Pa)' },
                  height: 150
                }}
                style={{ width: '100%', height: '150px' }}
              />
            </div>
          )}
        </Controls>
      </MainArea>
    </Container>
  );
};

export default App;
