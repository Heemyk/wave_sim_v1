import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Text } from '@react-three/drei';
import styled from 'styled-components';
import Plot from 'react-plotly.js';
import axios from 'axios';

// Types
interface SimulationConfig {
  room: {
    type: string;
    dimensions: number[];
    center: number[];
  };
  sources: Array<{
    id: string;
    position: number[];
    signal: {
      type: string;
      f0: number;
      f1: number;
    };
  }>;
  sensors: Array<{
    id: string;
    position: number[];
  }>;
  simulation: {
    fmin: number;
    fmax: number;
    df: number;
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
  config: SimulationConfig;
  sources: Array<{ id: string; position: number[] }>;
  sensors: Array<{ id: string; position: number[] }>;
}> = ({ config, sources, sensors }) => {
  return (
    <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      
      {/* Room */}
      <Box args={config.room.dimensions} position={config.room.center}>
        <meshBasicMaterial color="#e0e0e0" wireframe />
      </Box>
      
      {/* Sources */}
      {sources.map((source) => (
        <Sphere key={source.id} position={source.position} args={[0.1]}>
          <meshBasicMaterial color="#ff4444" />
        </Sphere>
      ))}
      
      {/* Sensors */}
      {sensors.map((sensor) => (
        <Sphere key={sensor.id} position={sensor.position} args={[0.05]}>
          <meshBasicMaterial color="#4444ff" />
        </Sphere>
      ))}
      
      <OrbitControls />
    </Canvas>
  );
};

// Main App Component
const App: React.FC = () => {
  const [config, setConfig] = useState<SimulationConfig>({
    room: {
      type: 'box',
      dimensions: [4, 3, 2.5],
      center: [0, 0, 0]
    },
    sources: [{
      id: 'source1',
      position: [0, 0, 1],
      signal: {
        type: 'chirp',
        f0: 20,
        f1: 8000
      }
    }],
    sensors: [{
      id: 'sensor1',
      position: [1, 1, 1]
    }],
    simulation: {
      fmin: 20,
      fmax: 2000,
      df: 50
    }
  });

  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [results, setResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection
  useEffect(() => {
    if (jobStatus?.job_id) {
      const ws = new WebSocket(`ws://localhost:8000/ws/${jobStatus.job_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setJobStatus(data);
        
        if (data.status === 'completed') {
          setIsRunning(false);
          fetchResults(data.job_id);
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
      const response = await axios.get(`http://localhost:8000/api/jobs/${jobId}/results`);
      setResults(response.data);
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  const runSimulation = async () => {
    try {
      setIsRunning(true);
      setJobStatus(null);
      setResults(null);

      const response = await axios.post('http://localhost:8000/api/simulate', config);
      setJobStatus(response.data);
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
            sensors={config.sensors}
          />
        </Viewer>
        
        <Controls>
          {results && (
            <div>
              <h3>Results</h3>
              <Plot
                data={[
                  {
                    x: results.frequencies?.map((f: any) => f.frequency) || [],
                    y: results.frequencies?.map((f: any) => 
                      Math.abs(f.sensor_data?.sensor1 || 0)
                    ) || [],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Frequency Response'
                  }
                ]}
                layout={{
                  title: 'Frequency Response',
                  xaxis: { title: 'Frequency (Hz)' },
                  yaxis: { title: 'Magnitude' },
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
