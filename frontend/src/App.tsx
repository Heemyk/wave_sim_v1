import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Text } from '@react-three/drei';
import styled from 'styled-components';
import Plot from 'react-plotly.js';
import axios from 'axios';

// No Electron dependencies - using direct web API calls

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
    target_h?: number;
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

// 3D Scene Component with FEM visualization
const Scene3D: React.FC<{
  config: SimulationRequest;
  sources: Array<{ id: string; position: number[] }>;
  sensors: Array<{ id: string; position: number[] }>;
  meshData?: any;
  fieldData?: any;
  selectedFrequency?: number;
}> = ({ config, sources, sensors, meshData, fieldData, selectedFrequency }) => {
  // Generate colors based on pressure field data
  const getVertexColors = () => {
    if (!fieldData?.field_data?.vertices || !selectedFrequency) {
      return null;
    }

    const vertices = fieldData.field_data.vertices;
    const pressures = fieldData.field_data.pressure_values || [];
    
    if (pressures.length !== vertices.length) {
      return null;
    }

    // Normalize pressure values for color mapping
    const magnitudes = pressures.map((p: any) => {
      if (typeof p === 'object' && p.real !== undefined) {
        return Math.sqrt(p.real * p.real + p.imag * p.imag);
      }
      return Math.abs(p || 0);
    });

    const minPressure = Math.min(...magnitudes);
    const maxPressure = Math.max(...magnitudes);
    const range = maxPressure - minPressure;

    // Create color array (RGB values for each vertex)
    const colors = new Float32Array(vertices.length * 3);
    
    for (let i = 0; i < magnitudes.length; i++) {
      const normalized = range > 0 ? (magnitudes[i] - minPressure) / range : 0.5;
      
      // Color mapping: blue (low) -> green -> yellow -> red (high)
      let r, g, b;
      if (normalized < 0.25) {
        // Blue to cyan
        r = 0;
        g = normalized * 4;
        b = 1;
      } else if (normalized < 0.5) {
        // Cyan to green
        r = 0;
        g = 1;
        b = 1 - (normalized - 0.25) * 4;
      } else if (normalized < 0.75) {
        // Green to yellow
        r = (normalized - 0.5) * 4;
        g = 1;
        b = 0;
      } else {
        // Yellow to red
        r = 1;
        g = 1 - (normalized - 0.75) * 4;
        b = 0;
      }
      
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
    
    return colors;
  };

  const vertexColors = getVertexColors();

  return (
    <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />
      
      {/* Room wireframe (always visible) */}
      <Box 
        args={[config.room.dimensions[0], config.room.dimensions[1], config.room.dimensions[2]]} 
        position={[config.room.center[0], config.room.center[1], config.room.center[2]]}
      >
        <meshBasicMaterial color="#888888" wireframe />
      </Box>
      
      {/* FEM Mesh with pressure field visualization */}
      {meshData?.mesh && fieldData?.field_data && (
        <group>
          {/* Render mesh vertices colored by pressure field */}
          {meshData.mesh.vertices?.map((vertex: number[], index: number) => {
            // Get pressure value for this vertex
            let pressure = 0;
            if (fieldData.field_data.pressure_values && fieldData.field_data.pressure_values[index]) {
              const p = fieldData.field_data.pressure_values[index];
              if (typeof p === 'object' && p.real !== undefined) {
                pressure = Math.sqrt(p.real * p.real + p.imag * p.imag);
              } else {
                pressure = Math.abs(p || 0);
              }
            }
            
            // Normalize pressure for color mapping (0-1)
            const maxPressure = 1.0; // You can adjust this based on your data
            const normalizedPressure = Math.min(pressure / maxPressure, 1.0);
            
            // Color mapping: blue (low) -> green -> yellow -> red (high)
            let hue;
            if (normalizedPressure < 0.25) {
              hue = 240 - normalizedPressure * 60; // Blue to cyan
            } else if (normalizedPressure < 0.5) {
              hue = 180 - (normalizedPressure - 0.25) * 60; // Cyan to green
            } else if (normalizedPressure < 0.75) {
              hue = 120 - (normalizedPressure - 0.5) * 60; // Green to yellow
            } else {
              hue = 60 - (normalizedPressure - 0.75) * 60; // Yellow to red
            }
            
            return (
              <Sphere
                key={index}
                position={[vertex[0], vertex[1], vertex[2]]}
                args={[0.05]} // Slightly larger for better visibility
              >
                <meshBasicMaterial 
                  color={`hsl(${hue}, 70%, 50%)`}
                />
              </Sphere>
            );
          })}
        </group>
      )}
      
      {/* Sources */}
      {sources.map((source) => (
        <group key={source.id}>
          <Sphere 
            position={[source.position[0], source.position[1], source.position[2]]} 
            args={[0.15]}
          >
            <meshBasicMaterial color="#ff4444" />
          </Sphere>
          <Text
            position={[source.position[0], source.position[1] + 0.3, source.position[2]]}
            fontSize={0.1}
            color="#ff4444"
            anchorX="center"
            anchorY="middle"
          >
            {source.id}
          </Text>
        </group>
      ))}
      
      {/* Sensors */}
      {sensors.map((sensor) => (
        <group key={sensor.id}>
          <Sphere 
            position={[sensor.position[0], sensor.position[1], sensor.position[2]]} 
            args={[0.08]}
          >
            <meshBasicMaterial color="#4444ff" />
          </Sphere>
          <Text
            position={[sensor.position[0], sensor.position[1] + 0.2, sensor.position[2]]}
            fontSize={0.08}
            color="#4444ff"
            anchorX="center"
            anchorY="middle"
          >
            {sensor.id}
          </Text>
        </group>
      ))}
      
      {/* Color bar legend */}
      {vertexColors && (
        <group position={[6, 0, 0]}>
          <Text
            position={[0, 3, 0]}
            fontSize={0.15}
            color="#000000"
            anchorX="center"
            anchorY="middle"
          >
            Pressure Field (Pa)
          </Text>
          {/* Color bar */}
          {Array.from({ length: 20 }, (_, i) => (
            <Box
              key={i}
              args={[0.1, 0.1, 0.1]}
              position={[0, 2.5 - i * 0.1, 0]}
            >
              <meshBasicMaterial color={
                (() => {
                  const normalized = i / 19;
                  if (normalized < 0.25) {
                    return `hsl(${240 - normalized * 60}, 100%, 50%)`;
                  } else if (normalized < 0.5) {
                    return `hsl(${180 - (normalized - 0.25) * 60}, 100%, 50%)`;
                  } else if (normalized < 0.75) {
                    return `hsl(${120 - (normalized - 0.5) * 60}, 100%, 50%)`;
                  } else {
                    return `hsl(${60 - (normalized - 0.75) * 60}, 100%, 50%)`;
                  }
                })()
              } />
            </Box>
          ))}
        </group>
      )}
      
      <OrbitControls enableDamping dampingFactor={0.05} />
    </Canvas>
  );
};

// Helper function for API calls
const apiCall = async (url: string, method: string = 'GET', data?: any) => {
  const baseUrl = 'http://localhost:8000';
  const fullUrl = url.startsWith('http') ? url : `${baseUrl}${url}`;
  
  return await axios({
    method,
    url: fullUrl,
    data,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    }
  });
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
      element_order: 1,
      target_h: 0.2  // Coarser mesh for faster computation
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
  const [selectedFrequency, setSelectedFrequency] = useState<number>(100);

  const wsRef = useRef<WebSocket | null>(null);

  // No Electron menu handlers needed for web app

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
           fetchFieldData(data.job_id, selectedFrequency);
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
      const newConfig = { ...prev } as any;
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

        {/* Frequency Selection for Visualization */}
        {results && results.frequencies && results.frequencies.length > 0 && (
          <FormGroup>
            <Label>Visualization Frequency</Label>
            <Select 
              value={selectedFrequency} 
              onChange={(e) => {
                const freq = parseFloat(e.target.value);
                setSelectedFrequency(freq);
                if (jobStatus?.job_id) {
                  fetchFieldData(jobStatus.job_id, freq);
                }
              }}
            >
              {results.frequencies.map((freq: any) => (
                <option key={freq.frequency} value={freq.frequency}>
                  {freq.frequency} Hz
                </option>
              ))}
            </Select>
          </FormGroup>
        )}

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
            selectedFrequency={selectedFrequency}
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
