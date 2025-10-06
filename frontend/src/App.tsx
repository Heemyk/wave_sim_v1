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
    type?: 'frequency_domain' | 'time_domain';
    duration?: number;
    sample_rate?: number;
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
  height: 300px;
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

// Frequency update function for unified data
const updateFrequencyVisualization = (
  unifiedData: any, 
  freqIndex: number, 
  availableFrequencies: number[], 
  setCurrentFrequencyIndex: (index: number) => void,
  setSelectedFrequency: (freq: number) => void,
  setFieldData: (data: any) => void,
  jobId?: string
) => {
  if (!unifiedData || !availableFrequencies[freqIndex]) return;
  
  const freq = availableFrequencies[freqIndex];
  setCurrentFrequencyIndex(freqIndex);
  setSelectedFrequency(freq);
  
  // Find the closest frequency key (since keys might be slightly different due to floating point precision)
  const freqKeys = Object.keys(unifiedData.frequency_data).map(k => parseFloat(k));
  console.log(`Looking for frequency ${freq} Hz, available keys:`, freqKeys.slice(0, 5), '...', freqKeys.slice(-5));
  const closestFreqKey = freqKeys.reduce((prev, curr) => 
    Math.abs(curr - freq) < Math.abs(prev - freq) ? curr : prev
  );
  console.log(`Closest frequency key found: ${closestFreqKey} Hz`);
  
  // Try multiple key formats to handle precision issues
  const possibleKeys = [
    closestFreqKey,
    Math.round(closestFreqKey),
    closestFreqKey.toString(),
    Math.round(closestFreqKey).toString(),
    `${closestFreqKey}.0`,  // Add .0 format
    `${Math.round(closestFreqKey)}.0`  // Add .0 format for rounded
  ];
  
  let foundData = null;
  let usedKey = null;
  
  for (const key of possibleKeys) {
    if (unifiedData.frequency_data[key] && unifiedData.frequency_data[key].pressure_field) {
      foundData = unifiedData.frequency_data[key];
      usedKey = key;
      break;
    }
  }
  
  console.log(`Tried keys: ${possibleKeys.join(', ')}, found data with key: ${usedKey}`);
  
  // Debug: Let's see what's actually in the frequency data
  if (usedKey === null) {
    console.log('Debugging frequency data structure:');
    console.log('Available keys:', Object.keys(unifiedData.frequency_data).slice(0, 5));
    console.log('Sample key 600:', unifiedData.frequency_data[600]);
    console.log('Sample key "600":', unifiedData.frequency_data["600"]);
    console.log('Sample key 600.0:', unifiedData.frequency_data[600.0]);
    console.log('Sample key "600.0":', unifiedData.frequency_data["600.0"]);
  }
  
  if (foundData) {
    setFieldData({
      field_data: {
        pressure_magnitude: foundData.pressure_field,
        pressure_real: foundData.pressure_field,
        pressure_imag: new Array(foundData.pressure_field?.length || 0).fill(0),
        frequency: usedKey
      },
      frequency: usedKey,
      job_id: jobId
    });
    console.log(`Updated frequency visualization to ${usedKey} Hz (requested ${freq} Hz)`);
  } else {
    console.warn(`No pressure field data for frequency ${freq} Hz`);
  }
};

// 3D Scene Component with FEM visualization
const Scene3D: React.FC<{
  config: SimulationRequest;
  sources: Array<{ id: string; position: number[] }>;
  sensors: Array<{ id: string; position: number[] }>;
  meshData?: any;
  fieldData?: any;
  selectedFrequency?: number;
  timeDomainData?: any;
  currentTimeStep?: number;
  visualizationMode?: 'frequency' | 'time';
}> = ({ config, sources, sensors, meshData, fieldData, selectedFrequency, timeDomainData, currentTimeStep, visualizationMode }) => {
  // Generate colors based on pressure field data (frequency domain)
  const getVertexColors = () => {
    console.log('🎨 getVertexColors called:', {
      timeDomainData: !!timeDomainData,
      currentTimeStep,
      fieldData: !!fieldData?.field_data,
      selectedFrequency,
      configType: config.simulation.type
    });
    
    // Use visualization mode to determine which colors to show
    if (visualizationMode === 'time' && timeDomainData && currentTimeStep !== undefined) {
      console.log('🕐 Using time domain colors');
      return getTimeDomainColors();
    }
    
    // Frequency domain mode
    if (visualizationMode === 'frequency' && fieldData?.field_data?.pressure_magnitude && selectedFrequency) {
      console.log('📊 Using frequency domain colors');
      return getFrequencyDomainColors();
    }
    
    // Time domain mode but no time domain data yet
    if (config.simulation.type === 'time_domain') {
      console.log('🕐 Time domain mode: No time domain data available yet. Click "Generate Time Domain" to compute it.');
      return null;
    }
    
    console.log('❌ No valid data for colors');
    return null;
  };

  // Generate colors for frequency domain visualization
  const getFrequencyDomainColors = () => {
    console.log('📊 getFrequencyDomainColors called with fieldData:', fieldData);
    
    // Try different field data structures based on your console logs
    let vertices, pressures;
    
    if (fieldData?.field_data?.pressure_magnitude && meshData?.vertices) {
      // Structure: fieldData.field_data.pressure_magnitude + meshData.vertices (direct from backend)
      vertices = meshData.vertices;
      pressures = fieldData.field_data.pressure_magnitude;
      console.log('📊 Using mesh + field_data structure:', {vertices: vertices?.length, pressures: pressures?.length});
    } else if (fieldData?.field_data?.pressure_magnitude && meshData?.mesh?.vertices) {
      // Structure: fieldData.field_data.pressure_magnitude + meshData.mesh.vertices (nested structure)
      vertices = meshData.mesh.vertices;
      pressures = fieldData.field_data.pressure_magnitude;
      console.log('📊 Using nested mesh + field_data structure:', {vertices: vertices?.length, pressures: pressures?.length});
    } else if (fieldData?.field_data?.vertices) {
      // Structure: fieldData.field_data.vertices
      vertices = fieldData.field_data.vertices;
      pressures = fieldData.field_data.pressure_values || fieldData.field_data.pressure_magnitude || [];
      console.log('📊 Using field_data structure:', {vertices: vertices?.length, pressures: pressures?.length});
    } else if (fieldData?.vertices) {
      // Structure: fieldData.vertices (direct access)
      vertices = fieldData.vertices;
      pressures = fieldData.pressure_values || fieldData.pressure_magnitude || [];
      console.log('📊 Using direct structure:', {vertices: vertices?.length, pressures: pressures?.length});
    } else if (meshData?.vertices && fieldData?.pressure_magnitude) {
      // Use mesh vertices with field pressure data (direct structure)
      vertices = meshData.vertices;
      pressures = fieldData.pressure_magnitude;
      console.log('📊 Using direct mesh + field structure:', {vertices: vertices?.length, pressures: pressures?.length});
    } else if (meshData?.mesh?.vertices && fieldData?.pressure_magnitude) {
      // Use mesh vertices with field pressure data (nested structure)
      vertices = meshData.mesh.vertices;
      pressures = fieldData.pressure_magnitude;
      console.log('📊 Using nested mesh + field structure:', {vertices: vertices?.length, pressures: pressures?.length});
    } else {
      console.log('❌ No valid field data structure found');
      console.log('🔍 Debug info:', {
        fieldData: !!fieldData,
        fieldDataFieldData: !!fieldData?.field_data,
        fieldDataPressureMagnitude: !!fieldData?.field_data?.pressure_magnitude,
        meshData: !!meshData,
        meshVertices: !!meshData?.mesh?.vertices
      });
      
      // Debug: Let's see the actual mesh data structure
      console.log('🔍 Mesh data structure:', meshData);
      console.log('🔍 Mesh data keys:', meshData ? Object.keys(meshData) : 'no mesh data');
      if (meshData?.mesh) {
        console.log('🔍 Mesh.mesh keys:', Object.keys(meshData.mesh));
      }
      if (meshData?.vertices) {
        console.log('🔍 Mesh.vertices available:', !!meshData.vertices);
      }
      return null;
    }
    
    if (!vertices || !pressures || pressures.length !== vertices.length) {
      console.log('❌ Data length mismatch:', {vertices: vertices?.length, pressures: pressures?.length});
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

    console.log(`📊 Pressure range: ${minPressure.toExponential(2)} to ${maxPressure.toExponential(2)}`);

    // Use logarithmic scaling for better visualization of small values
    const logMagnitudes = magnitudes.map((m: number) => m > 0 ? Math.log10(m) : -20); // Cap at -20 for zero values
    const logMin = Math.min(...logMagnitudes);
    const logMax = Math.max(...logMagnitudes);
    const logRange = logMax - logMin;

    console.log(`📊 Log pressure range: ${logMin.toExponential(2)} to ${logMax.toExponential(2)}`);

    // Create color array (RGB values for each vertex)
    const colors = new Float32Array(vertices.length * 3);
    
    for (let i = 0; i < magnitudes.length; i++) {
      // Use logarithmic normalization for better color distribution
      const normalized = logRange > 0 ? (logMagnitudes[i] - logMin) / logRange : 0.5;
      
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
    
    console.log('✅ Frequency domain colors generated successfully');
    console.log('🎨 Color array length:', colors.length);
    console.log('🎨 First few color values:', Array.from(colors.slice(0, 12))); // First 4 vertices (12 RGB values)
    console.log('🎨 Last few color values:', Array.from(colors.slice(-12))); // Last 4 vertices (12 RGB values)
    return colors;
  };

  // Generate colors for time-domain visualization
  const getTimeDomainColors = () => {
    console.log('🕐 getTimeDomainColors called:', {
      timeDomainData: !!timeDomainData,
      timeFieldData: !!timeDomainData?.time_field_data,
      currentTimeStep,
      timeSteps: timeDomainData?.time_field_data?.time_steps?.length,
      pressureTimeSeries: timeDomainData?.time_field_data?.pressure_time_series?.length,
      meshCoords: timeDomainData?.time_field_data?.mesh_coordinates?.length,
      timeDomainDataKeys: timeDomainData ? Object.keys(timeDomainData) : 'no timeDomainData',
      timeFieldDataKeys: timeDomainData?.time_field_data ? Object.keys(timeDomainData.time_field_data) : 'no time_field_data'
    });
    
    // Debug: Check if timeDomainData has the mesh data but no time_field_data
    if (timeDomainData && !timeDomainData.time_field_data) {
      console.log('🔍 DIAGNOSIS: timeDomainData exists but time_field_data is missing');
      console.log('🔍 Available keys:', Object.keys(timeDomainData));
      console.log('🔍 Has mesh_pressure_history:', !!timeDomainData.mesh_pressure_history);
      console.log('🔍 Has mesh_coordinates:', !!timeDomainData.mesh_coordinates);
      console.log('🔍 Has mesh_time_steps:', !!timeDomainData.mesh_time_steps);
      
      // FIX: Create the missing time_field_data structure on-demand
      console.log('🔧 FIXING: Creating time_field_data structure from raw data');
      timeDomainData.time_field_data = {
        pressure_time_series: timeDomainData.mesh_pressure_history || [],
        mesh_coordinates: timeDomainData.mesh_coordinates || [],
        time_steps: timeDomainData.mesh_time_steps || timeDomainData.time_vector || [],
        num_time_steps: timeDomainData.mesh_time_steps?.length || timeDomainData.num_time_steps || 0
      };
      console.log('✅ Created time_field_data structure:', timeDomainData.time_field_data);
    }
    
    if (!timeDomainData?.time_field_data || currentTimeStep === undefined) {
      console.log('❌ Missing time domain data or time step');
      console.log('  timeDomainData exists:', !!timeDomainData);
      console.log('  time_field_data exists:', !!timeDomainData?.time_field_data);
      console.log('  currentTimeStep:', currentTimeStep);
      return null;
    }
    
    const pressureTimeSeries = timeDomainData.time_field_data.pressure_time_series;
    const meshCoords = timeDomainData.time_field_data.mesh_coordinates;
    
    // NEW: pressureTimeSeries is now an array of pressure arrays (one per time step)
    if (!pressureTimeSeries || !Array.isArray(pressureTimeSeries) || currentTimeStep >= pressureTimeSeries.length) {
      console.log('❌ Invalid time domain data structure - pressureTimeSeries should be array of arrays');
      console.log('  pressureTimeSeries type:', typeof pressureTimeSeries, 'length:', pressureTimeSeries?.length);
      return null;
    }
    
    // Get pressure values for current time step
    const pressures = pressureTimeSeries[currentTimeStep];
    
    if (!pressures || !Array.isArray(pressures) || pressures.length !== meshCoords.length) {
      console.log('❌ Pressure data length mismatch');
      console.log('  pressures length:', pressures?.length, 'meshCoords length:', meshCoords?.length);
      return null;
    }
    
    // Normalize pressure values
    const minPressure = Math.min(...pressures);
    const maxPressure = Math.max(...pressures);
    const pressureRange = maxPressure - minPressure;
    
    console.log(`🕐 Time step ${currentTimeStep}: pressure range ${minPressure.toExponential(2)} to ${maxPressure.toExponential(2)}`);
    
    // Create color array
    const colors = new Float32Array(meshCoords.length * 3);
    
    for (let i = 0; i < pressures.length; i++) {
      let normalizedPressure = 0;
      if (pressureRange > 0) {
        normalizedPressure = (pressures[i] - minPressure) / pressureRange;
      }
      
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
      
      // Convert HSL to RGB (simplified)
      const saturation = 0.8;
      const lightness = 0.6;
      const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
      const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
      const m = lightness - c / 2;
      
      let r, g, b;
      if (hue < 60) {
        r = c; g = x; b = 0;
      } else if (hue < 120) {
        r = x; g = c; b = 0;
      } else if (hue < 180) {
        r = 0; g = c; b = x;
      } else if (hue < 240) {
        r = 0; g = x; b = c;
      } else if (hue < 300) {
        r = x; g = 0; b = c;
      } else {
        r = c; g = 0; b = x;
      }
      
      colors[i * 3] = r + m;
      colors[i * 3 + 1] = g + m;
      colors[i * 3 + 2] = b + m;
    }
    
    console.log('✅ Time domain colors generated successfully');
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
      {(meshData?.vertices || meshData?.mesh || timeDomainData?.time_field_data) && (fieldData?.field_data || timeDomainData) && (
        <group>
          {/* Render mesh vertices colored by pressure field */}
          {(() => {
            console.log('🎨 Rendering mesh with vertex colors:', !!vertexColors);
            
            // Use time domain mesh coordinates if available, otherwise use frequency domain
            const vertices = timeDomainData?.time_field_data?.mesh_coordinates || meshData?.vertices || meshData?.mesh?.vertices;
            
            if (!vertices || !vertexColors) {
              console.log('❌ No vertices or colors available');
              return null;
            }
            
            console.log(`🎨 Rendering ${vertices.length} vertices with colors`);
            
            return vertices.map((vertex: number[], index: number) => {
              if (index * 3 + 2 >= vertexColors.length) {
                console.log(`❌ Color index ${index} out of range`);
                return null;
              }
              
              const r = vertexColors[index * 3];
              const g = vertexColors[index * 3 + 1];
              const b = vertexColors[index * 3 + 2];
              
              return (
                <Sphere
                  key={index}
                  position={[vertex[0], vertex[1], vertex[2]]}
                  args={[0.05]}
                >
                  <meshBasicMaterial 
                    color={`rgb(${Math.floor(r * 255)}, ${Math.floor(g * 255)}, ${Math.floor(b * 255)})`}
                  />
                </Sphere>
              );
            });
          })()}
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
  
  // Log all API calls
  console.log(`🚀 API CALL: ${method} ${fullUrl}`);
  if (data) {
    console.log('📤 Request data:', data);
  }
  
  const startTime = Date.now();
  
  try {
    const response = await axios({
      method,
      url: fullUrl,
      data,
      timeout: 900000, // 15 minute timeout for unified simulations
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    const duration = Date.now() - startTime;
    console.log(`✅ API SUCCESS: ${method} ${fullUrl} (${duration}ms)`);
    console.log('📥 Response data:', response.data);
    
    return response;
  } catch (error: any) {
    const duration = Date.now() - startTime;
    console.error(`❌ API ERROR: ${method} ${fullUrl} (${duration}ms)`);
    console.error('Error details:', error.response?.data || error.message);
    throw error;
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
      element_order: 1,
      target_h: 0.2  // Coarser mesh for faster computation
    },
    simulation: {
      type: 'frequency_domain',  // Default to frequency domain
      fmin: 100,
      fmax: 200,
      df: 100,
      duration: 2.0,
      sample_rate: 44100
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
  const [impulseResponse, setImpulseResponse] = useState<any>(null);
  const [selectedSensorIndex, setSelectedSensorIndex] = useState<number>(0);
  const [audioGain, setAudioGain] = useState<number>(1.0);
  const [isComputingAudio, setIsComputingAudio] = useState(false);
  const [timeDomainData, setTimeDomainData] = useState<any>(null);
  const [currentTimeStep, setCurrentTimeStep] = useState<number>(0);
  const [isPlayingAnimation, setIsPlayingAnimation] = useState(false);
  // const [timeDomainChunkInfo, setTimeDomainChunkInfo] = useState<any>(null); // Removed unused variable
  const [unifiedData, setUnifiedData] = useState<any>(null);
  const [visualizationMode, setVisualizationMode] = useState<'frequency' | 'time'>('frequency');
  const [currentFrequencyIndex, setCurrentFrequencyIndex] = useState<number>(0);
  const [availableFrequencies, setAvailableFrequencies] = useState<number[]>([]);
  const [customAudioFile, setCustomAudioFile] = useState<File | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastFetchRef = useRef<{results: number, mesh: number}>({results: 0, mesh: 0});
  const lastCompletionRef = useRef<string | null>(null);
  const lastRenderTimeRef = useRef<number>(0);

  // Animation loop for time-domain visualization
  useEffect(() => {
    if (isPlayingAnimation && timeDomainData?.time_field_data) {
      const maxSteps = timeDomainData.time_field_data.num_time_steps - 1;
      
      const animate = () => {
        setCurrentTimeStep(prev => {
          const next = prev + 1;
          return next > maxSteps ? 0 : next; // Loop back to start
        });
        animationRef.current = requestAnimationFrame(animate);
      };
      
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlayingAnimation, timeDomainData]);

  // No Electron menu handlers needed for web app

  // WebSocket connection
  useEffect(() => {
    // Only create WebSocket if we have a job_id and don't already have a connection
    if (jobStatus?.job_id && !wsRef.current) {
      console.log('🔌 Creating WebSocket connection for job:', jobStatus.job_id);
      const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobStatus.job_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Throttle WebSocket message processing to prevent excessive re-renders
        const now = Date.now();
        if (now - lastRenderTimeRef.current < 500) { // Throttle to max 2 calls per second
          console.log('⏭️ Throttling WebSocket message processing');
          return;
        }
        lastRenderTimeRef.current = now;
        
        console.log('📨 WebSocket message received:', data);
        setJobStatus(data);
        
        if (data.status === 'completed') {
           // Prevent duplicate processing of completion messages
           if (lastCompletionRef.current === data.job_id) {
             console.log('⏭️ Skipping duplicate completion message for job:', data.job_id);
             return;
           }
           
           console.log('✅ Job completed, fetching data...');
           console.log('🔧 Simulation type:', config.simulation.type);
           lastCompletionRef.current = data.job_id;
          setIsRunning(false);
          fetchResults(data.job_id);
          fetchMeshData(data.job_id);
          
           // Fetch appropriate data based on simulation type
           if (config.simulation.type === 'time_domain') {
             console.log('🕐 Time domain mode: NOT fetching frequency field data');
             // Time domain data will be fetched when user clicks "Generate Time Domain"
           } else {
             console.log('📊 Frequency domain mode: fetching field data');
             fetchFieldData(data.job_id, selectedFrequency);
           }
        } else if (data.status === 'failed') {
          console.log('❌ Job failed');
          setIsRunning(false);
        }
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket connection closed');
        wsRef.current = null;
      };

      ws.onerror = (error) => {
        console.error('🔌 WebSocket error:', error);
      };
    }

    // Cleanup function
    return () => {
      console.log('🔌 Cleaning up WebSocket connection');
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [jobStatus?.job_id]); // Only recreate when job_id actually changes

  const fetchResults = async (jobId: string) => {
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchRef.current.results;
    
    // Debounce: only fetch if it's been more than 5 seconds since last fetch
    if (timeSinceLastFetch < 5000) {
      console.log('📊 Skipping results fetch (debounced)');
      return;
    }
    
    try {
      console.log('📊 Fetching results for job:', jobId);
      lastFetchRef.current.results = now;
      const response = await apiCall(`/api/jobs/${jobId}/results`);
      setResults(response.data);
      console.log('📊 Results fetched successfully');
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  const fetchMeshData = async (jobId: string) => {
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchRef.current.mesh;
    
    // Debounce: only fetch if it's been more than 5 seconds since last fetch
    if (timeSinceLastFetch < 5000) {
      console.log('🔲 Skipping mesh fetch (debounced)');
      return;
    }
    
    try {
      console.log('🔲 Fetching mesh data for job:', jobId);
      lastFetchRef.current.mesh = now;
      const response = await apiCall(`/api/jobs/${jobId}/mesh`);
      setMeshData(response.data);
      console.log('🔲 Mesh data fetched successfully');
    } catch (error) {
      console.error('Error fetching mesh data:', error);
    }
  };

  const fetchFieldData = async (jobId: string, frequency: number) => {
    try {
      console.log(`Fetching field data for job ${jobId} at frequency ${frequency}`);
      const response = await apiCall(`/api/jobs/${jobId}/field/${frequency}`);
      console.log('Field data response:', response.data);
      setFieldData(response.data);
    } catch (error) {
      console.error('Error fetching field data:', error);
    }
  };

  const computeTimeDomainSimulation = async () => {
    setIsComputingAudio(true);
    
    try {
      console.log('Computing time-domain wave equation simulation for realistic echoes');
      
      // Process audio file if available
      let audioData: Float32Array | null = null;
      let audioDuration = config.simulation.duration || 1.0;
      let audioSampleRate = config.simulation.sample_rate || 44100;
      
      if (customAudioFile) {
        console.log('Processing audio file for time-domain simulation:', customAudioFile.name);
        audioData = await processAudioFile(customAudioFile, audioSampleRate);
        audioDuration = audioData.length / audioSampleRate; // Use actual audio duration
        console.log(`Audio file processed: ${audioData.length} samples, duration: ${audioDuration}s`);
      } else {
        console.log('No custom audio file, creating default sine wave');
        // Create a short impulse-like signal for testing echoes
        const numSamples = Math.floor(0.1 * audioSampleRate); // 0.1 second impulse
        audioData = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
          audioData[i] = Math.sin(2 * Math.PI * 440 * i / audioSampleRate) * Math.exp(-i / (numSamples / 10));
        }
        audioDuration = 0.1;
      }
      
      const request = {
        sensor_positions: config.output.sensors.map((s: any) => s.position),
        source_position: config.sources[0]?.position || [1.0, 1.0, 1.0],
        sample_rate: audioSampleRate,
        duration: audioDuration,
        mesh_file: "data/meshes/box_Frontend Test Simulation.msh",
        element_order: 1,
        boundary_impedance: {
          walls: config.boundaries.walls.alpha,
          floor: config.boundaries.floor.alpha,
          ceiling: config.boundaries.ceiling.alpha
        },
        source_signal: audioData ? Array.from(audioData) : null
      };
      
      console.log('Sending time-domain simulation request:', {
        sensorCount: request.sensor_positions.length,
        duration: request.duration,
        sampleRate: request.sample_rate,
        sourceSignalLength: request.source_signal?.length,
        boundaryImpedance: request.boundary_impedance
      });
      
      const response = await apiCall('/api/time_domain_simulation', 'POST', request);
      console.log('🎯 TIME DOMAIN API RESPONSE:', response.data);
      console.log('🎯 RESPONSE SUCCESS:', response.data?.success);
      console.log('🎯 HAS TIME DOMAIN DATA:', !!response.data?.time_domain_data);
      
      if (response.data?.time_domain_data) {
        const timeData = response.data.time_domain_data;
        console.log('🎯 RECEIVED TIME DOMAIN DATA FROM BACKEND:', timeData);
        
        // The NEW time-domain solver directly provides pressure over time - no impulse responses needed!
        // sensor_data contains the actual pressure values at each time step
        const sensorTimeSeriesData: {[key: number]: any} = {};
        for (const [sensorIdx, pressureTimeSeries] of Object.entries(timeData.sensor_data)) {
          sensorTimeSeriesData[parseInt(sensorIdx)] = {
            pressure_time_series: pressureTimeSeries, // Direct pressure over time
            sample_rate: timeData.sample_rate,
            duration: timeData.duration,
            time_vector: timeData.time_vector
          };
        }
        
        // Debug: Check what data we received from backend
        console.log('🔍 Backend time domain data structure:', {
          hasMeshPressureHistory: !!timeData.mesh_pressure_history,
          meshPressureHistoryLength: timeData.mesh_pressure_history?.length,
          hasMeshCoordinates: !!timeData.mesh_coordinates,
          meshCoordinatesLength: timeData.mesh_coordinates?.length,
          hasMeshTimeSteps: !!timeData.mesh_time_steps,
          meshTimeStepsLength: timeData.mesh_time_steps?.length,
          fullTimeDataKeys: Object.keys(timeData)
        });
        
        // Set up time domain data with direct time-domain solution
        const formattedTimeData = {
          ...timeData,
          sensor_time_series: sensorTimeSeriesData, // Direct pressure data over time
          custom_source_signal: request.source_signal,
          // Add fields for visualization compatibility
          time_field_data: {
            pressure_time_series: timeData.mesh_pressure_history || [], // Full mesh pressure over time
            mesh_coordinates: timeData.mesh_coordinates || [], // Mesh node coordinates
            time_steps: timeData.mesh_time_steps || timeData.time_vector || [],
            num_time_steps: timeData.mesh_time_steps?.length || timeData.num_time_steps || 0
          },
          parameters: {
            duration: timeData.duration,
            num_time_steps: timeData.num_time_steps,
            sample_rate: timeData.sample_rate
          }
        };
        
        console.log('🔍 Formatted time domain data structure:', {
          hasTimeFieldData: !!formattedTimeData.time_field_data,
          hasPressureTimeSeries: !!formattedTimeData.time_field_data?.pressure_time_series,
          pressureTimeSeriesLength: formattedTimeData.time_field_data?.pressure_time_series?.length,
          hasMeshCoordinates: !!formattedTimeData.time_field_data?.mesh_coordinates,
          meshCoordinatesLength: formattedTimeData.time_field_data?.mesh_coordinates?.length,
          hasTimeSteps: !!formattedTimeData.time_field_data?.time_steps,
          timeStepsLength: formattedTimeData.time_field_data?.time_steps?.length
        });
        
        console.log('🎯 SETTING TIME DOMAIN DATA:', formattedTimeData);
        console.log('🎯 TIME_FIELD_DATA STRUCTURE:', formattedTimeData.time_field_data);
        setTimeDomainData(formattedTimeData);
        console.log('✅ Time-domain simulation completed successfully');
        console.log('Sensor data available for', Object.keys(timeData.sensor_data).length, 'sensors');
        console.log('Time steps:', timeData.num_time_steps);
        
        // Also set unifiedData to show the same UI as unified simulation
        setUnifiedData({
          frequency_data: {}, // Empty for time-domain only
          time_domain_data: formattedTimeData
        });
        
        // Play the result immediately to hear the echoes
        console.log('🎵 Time-domain simulation completed - ready to play echoes');
        // await playAudio(); // Commented out to avoid auto-play
        
      } else {
        console.error('❌ No time-domain data found in response');
      }
      
        } catch (error: any) {
          console.error('❌ ERROR in time-domain simulation:', error);
          console.error('❌ Error details:', error.response?.data || error.message);
        } finally {
          setIsComputingAudio(false);
        }
  };

  const computeUnifiedSimulation = async () => {
    setIsComputingAudio(true);
    
    try {
      console.log('Computing unified simulation (frequency + time domain)');
      
      // Process audio file if available
      let audioData: Float32Array | null = null;
      let audioDuration = config.simulation.duration || 1.0;
      let audioSampleRate = config.simulation.sample_rate || 44100;
      
      if (customAudioFile) {
        console.log('Processing audio file for simulation:', customAudioFile.name);
        audioData = await processAudioFile(customAudioFile, audioSampleRate);
        audioDuration = audioData.length / audioSampleRate; // Use actual audio duration
        console.log(`Audio file processed: ${audioData.length} samples, duration: ${audioDuration}s`);
      } else {
        console.log('No custom audio file, using default sine wave');
      }
      
      const request = {
        sensor_positions: config.output.sensors.map((s: any) => s.position),
        source_position: config.sources[0]?.position || [1.0, 1.0, 1.0],
        source_frequency: 440, // Fallback frequency for frequency domain analysis
        sample_rate: audioSampleRate,
        duration: audioDuration, // Use actual audio duration
        max_frequency: audioSampleRate / 2, // Use Nyquist frequency
        num_frequencies: 100,  // Compute 100 frequencies
        mesh_file: "data/meshes/box_Frontend Test Simulation.msh", // Default mesh
        element_order: 1,
        boundary_impedance: {
          walls: config.boundaries.walls.alpha,
          floor: config.boundaries.floor.alpha,
          ceiling: config.boundaries.ceiling.alpha
        },
        // Send processed audio data
        custom_audio_data: audioData ? Array.from(audioData) : null,
        custom_audio_filename: customAudioFile?.name || null,
        use_custom_audio: !!customAudioFile
      };
      
      const response = await apiCall('/api/unified_simulation', 'POST', request);
      console.log('Unified simulation response:', response.data);
      
      let unifiedSim;
      
      // Check if data was saved to file due to large size
      if (response.data?.unified_file) {
        console.log('Data saved to file, fetching from file endpoint...');
        const fileResponse = await apiCall(`/api/jobs/${response.data.job_id}/unified_data`, 'GET');
        console.log('File response received:', {
          hasData: !!fileResponse.data,
          dataType: typeof fileResponse.data,
          dataKeys: fileResponse.data ? Object.keys(fileResponse.data) : 'no data'
        });
        unifiedSim = fileResponse.data?.unified_data;
      } else {
        // Direct response
        unifiedSim = response.data?.unified_data;
      }
      
      if (!unifiedSim) {
        console.error('No unified simulation data found:', response.data);
        return;
      }
      
      console.log('Unified simulation data loaded:', {
        frequency_data: Object.keys(unifiedSim.frequency_data || {}),
        time_domain_data: !!unifiedSim.time_domain_data,
        parameters: unifiedSim.parameters
      });
      
      setUnifiedData(unifiedSim);
      
      // Extract frequency list for slider
      const frequencies = Object.keys(unifiedSim.frequency_data || {})
        .map(f => parseFloat(f))
        .sort((a, b) => a - b);
      console.log('Available frequencies from data:', frequencies.slice(0, 10), '...', frequencies.slice(-10));
      setAvailableFrequencies(frequencies);
      setCurrentFrequencyIndex(0);
      
      // Set frequency domain data for current visualization
      const firstFreq = frequencies[0];
      if (firstFreq) {
        // Find the actual key in the frequency data
        const freqKeys = Object.keys(unifiedSim.frequency_data).map(k => parseFloat(k));
        const closestFirstFreq = freqKeys.reduce((prev, curr) => 
          Math.abs(curr - firstFreq) < Math.abs(prev - firstFreq) ? curr : prev
        );
        
        if (unifiedSim.frequency_data[closestFirstFreq] && unifiedSim.frequency_data[closestFirstFreq].pressure_field) {
          setFieldData({
            field_data: {
              pressure_magnitude: unifiedSim.frequency_data[closestFirstFreq].pressure_field,
              pressure_real: unifiedSim.frequency_data[closestFirstFreq].pressure_field,
              pressure_imag: new Array(unifiedSim.frequency_data[closestFirstFreq].pressure_field?.length || 0).fill(0),
              frequency: closestFirstFreq
            },
            frequency: closestFirstFreq,
            job_id: response.data.job_id || 'unified-simulation'
          });
        }
      }
      
      // Set time domain data with proper formatting for visualization
      const unifiedTimeData = unifiedSim.time_domain_data;
      
      // Convert sensor data to audio format for unified simulation
      const sensorTimeSeriesData: {[key: number]: any} = {};
      if (unifiedTimeData.sensor_data) {
        for (const [sensorIdx, pressureTimeSeries] of Object.entries(unifiedTimeData.sensor_data)) {
          sensorTimeSeriesData[parseInt(sensorIdx)] = {
            pressure_time_series: pressureTimeSeries,
            sample_rate: unifiedTimeData.sample_rate,
            duration: unifiedTimeData.duration,
            time_vector: unifiedTimeData.time_vector
          };
        }
      }
      
      // Format the unified time domain data to include time_field_data structure
      const formattedUnifiedTimeData = {
        ...unifiedTimeData,
        sensor_time_series: sensorTimeSeriesData, // Add sensor audio data for unified simulation
        time_field_data: {
          pressure_time_series: unifiedTimeData.mesh_pressure_history || [],
          mesh_coordinates: unifiedTimeData.mesh_coordinates || [],
          time_steps: unifiedTimeData.mesh_time_steps || unifiedTimeData.time_vector || [],
          num_time_steps: unifiedTimeData.mesh_time_steps?.length || unifiedTimeData.num_time_steps || 0
        }
      };
      
      console.log('🎯 SETTING UNIFIED TIME DOMAIN DATA:', formattedUnifiedTimeData);
      console.log('🎯 UNIFIED TIME_FIELD_DATA STRUCTURE:', formattedUnifiedTimeData.time_field_data);
      
      setTimeDomainData(formattedUnifiedTimeData);
      
      // Extract time series data for the first sensor
      if (unifiedSim.time_domain_data?.sensor_time_series?.[0]) {
        console.log('Time series data available for first sensor');
        console.log('Pressure time series length:', unifiedSim.time_domain_data.sensor_time_series[0].pressure_time_series.length);
      }
      
      // Set mesh data from unified response (no need to fetch separately!)
      if (unifiedSim.mesh_data) {
        console.log('Setting mesh data from unified response:', unifiedSim.mesh_data);
        setMeshData(unifiedSim.mesh_data);
      } else {
        console.warn('No mesh data found in unified response');
      }
      
    } catch (error) {
      console.error('Error computing unified simulation:', error);
    } finally {
      setIsComputingAudio(false);
    }
  };


  // Audio file processing function
  const processAudioFile = async (file: File, targetSampleRate: number): Promise<Float32Array> => {
    return new Promise((resolve, reject) => {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const fileReader = new FileReader();
      
      fileReader.onload = async (e) => {
        try {
          const arrayBuffer = e.target?.result as ArrayBuffer;
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          
          console.log(`Original audio: ${audioBuffer.sampleRate}Hz, ${audioBuffer.duration}s, ${audioBuffer.numberOfChannels} channels`);
          
          // Get the first channel (mono)
          const sourceData = audioBuffer.getChannelData(0);
          
          // Resample if needed
          let processedData: Float32Array;
          if (audioBuffer.sampleRate !== targetSampleRate) {
            console.log(`Resampling from ${audioBuffer.sampleRate}Hz to ${targetSampleRate}Hz`);
            processedData = resampleAudio(sourceData, audioBuffer.sampleRate, targetSampleRate);
          } else {
            processedData = sourceData;
          }
          
          console.log(`Processed audio: ${processedData.length} samples`);
          resolve(processedData);
        } catch (error) {
          reject(error);
        }
      };
      
      fileReader.onerror = () => reject(new Error('Failed to read audio file'));
      fileReader.readAsArrayBuffer(file);
    });
  };
  
  // Simple resampling function (linear interpolation)
  const resampleAudio = (inputData: Float32Array, inputSampleRate: number, outputSampleRate: number): Float32Array => {
    const ratio = inputSampleRate / outputSampleRate;
    const outputLength = Math.floor(inputData.length / ratio);
    const output = new Float32Array(outputLength);
    
    for (let i = 0; i < outputLength; i++) {
      const inputIndex = i * ratio;
      const index = Math.floor(inputIndex);
      const fraction = inputIndex - index;
      
      if (index + 1 < inputData.length) {
        // Linear interpolation
        output[i] = inputData[index] * (1 - fraction) + inputData[index + 1] * fraction;
      } else {
        output[i] = inputData[index] || 0;
      }
    }
    
    return output;
  };

  const playAudio = async () => {
    if (!timeDomainData?.sensor_time_series) {
      console.log('No time-domain sensor data available');
      return;
    }

    try {
      console.log('Playing audio with direct time-domain data for sensor', selectedSensorIndex);
      
      // Get the pressure time series for the selected sensor
      const sensorTimeSeries = timeDomainData.sensor_time_series[selectedSensorIndex];
      if (!sensorTimeSeries) {
        console.error(`No time series data found for sensor ${selectedSensorIndex}`);
        return;
      }
      
      console.log('🎵 DIRECT TIME-DOMAIN AUDIO DEBUG INFO:');
      console.log('  📍 Selected sensor:', selectedSensorIndex, config.output.sensors[selectedSensorIndex]?.id);
      console.log('  📍 Sensor position:', config.output.sensors[selectedSensorIndex]?.position);
      console.log('  🔊 Pressure time series length:', sensorTimeSeries.pressure_time_series.length);
      const pressureArray = Array.from(sensorTimeSeries.pressure_time_series as number[]);
      console.log('  📊 Pressure signal max:', Math.max(...pressureArray.map(Math.abs)));
      console.log('  📊 Pressure signal min:', Math.min(...pressureArray));
      console.log('  🎚️ Audio gain:', audioGain);
      
      // The pressure time series IS the audio signal - no convolution needed!
      // This is the actual pressure fluctuation at the sensor over time
      const pressureSignal = new Float32Array(sensorTimeSeries.pressure_time_series);
      
      // DEBUG: Check if the signal looks like speech or just noise
      console.log('🔍 AUDIO SIGNAL ANALYSIS:');
      console.log('  Signal length:', pressureSignal.length);
      console.log('  Signal range:', Math.min(...Array.from(pressureSignal)), 'to', Math.max(...Array.from(pressureSignal)));
      console.log('  Signal RMS:', Math.sqrt(pressureSignal.reduce((sum, val) => sum + val*val, 0) / pressureSignal.length));
      
      // Check for signal characteristics that indicate speech vs noise
      const nonZeroSamples = Array.from(pressureSignal).filter(x => Math.abs(x) > 1e-10).length;
      console.log('  Non-zero samples:', nonZeroSamples, '/', pressureSignal.length, `(${(nonZeroSamples/pressureSignal.length*100).toFixed(1)}%)`);
      
      // Check if signal is mostly zeros (would sound like a pop)
      if (nonZeroSamples < pressureSignal.length * 0.01) {
        console.log('⚠️  WARNING: Signal is mostly zeros - this will sound like a pop!');
        console.log('    This suggests the simulation is not generating proper wave propagation.');
      } else {
        console.log('✅ Signal has meaningful content - should sound like speech after normalization.');
      }
      
      // Check if signal amplitude is too small for audio
      const signalRMS = Math.sqrt(pressureSignal.reduce((sum, val) => sum + val*val, 0) / pressureSignal.length);
      if (signalRMS < 1e-6) {
        console.log('⚠️  WARNING: Signal amplitude is very small (RMS < 1e-6) - will need significant amplification.');
      } else {
        console.log(`✅ Signal amplitude is reasonable (RMS = ${signalRMS.toExponential(2)}).`);
      }
      
      // Normalize and apply gain
      const normalizedSignal = normalizeSignal(pressureSignal);
      const gainAdjustedSignal = new Float32Array(normalizedSignal.length);
      for (let i = 0; i < normalizedSignal.length; i++) {
        gainAdjustedSignal[i] = normalizedSignal[i] * audioGain;
      }
      
      console.log(`Normalized signal max: ${Math.max(...Array.from(normalizedSignal.map(Math.abs)))}`);
      console.log(`Applied gain: ${audioGain}x`);
      
      const sampleRate = sensorTimeSeries.sample_rate || 44100;
      
      console.log('🎵 Playing simulation result with improved normalization...');
      await playAudioBuffer(gainAdjustedSignal, sampleRate);
      console.log('🎵 Audio playback completed - this is the ACTUAL pressure at the sensor position!');
      console.log('   No impulse response or convolution needed - direct time-domain solution!');
      
    } catch (error) {
      console.error('Error playing audio:', error);
    }
  };

  // convolveSignals function removed - no longer needed with direct time-domain solver!

  const normalizeSignal = (signal: Float32Array): Float32Array => {
    const max = Math.max(...Array.from(signal.map(Math.abs)));
    if (max === 0) return signal;
    
    // For acoustic pressure signals, we need to scale to audible levels
    // Target amplitude should be around 0.1 to 0.5 for good audio
    const targetAmplitude = 0.3; // Much larger than the tiny acoustic pressures
    const scaleFactor = targetAmplitude / max;
    
    console.log(`🔊 Normalizing signal: max=${max.toExponential(2)}, scale factor=${scaleFactor.toExponential(2)}`);
    
    return new Float32Array(signal.map(s => s * scaleFactor));
  };

  const playAudioBuffer = async (signal: Float32Array, sampleRate: number): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const audioBuffer = audioContext.createBuffer(1, signal.length, sampleRate);
        audioBuffer.copyToChannel(new Float32Array(signal), 0);
        
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        
        source.onended = () => resolve();
        source.start();
        
      } catch (error) {
        reject(error);
      }
    });
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
      
      // Reset completion tracking for new job
      lastCompletionRef.current = null;

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
           <Label>Simulation Type</Label>
           <Select 
             value={config.simulation.type || 'frequency_domain'} 
             onChange={(e) => updateConfig('simulation.type', e.target.value)}
           >
             <option value="frequency_domain">Frequency Domain (Steady State)</option>
             <option value="time_domain">Time Domain (Wave Propagation)</option>
           </Select>
         </FormGroup>

         {config.simulation.type === 'frequency_domain' ? (
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
         ) : (
           <FormGroup>
             <Label>Time Domain Settings</Label>
             <div style={{ display: 'flex', gap: '5px', flexDirection: 'column' }}>
               <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
                 <span style={{ minWidth: '80px' }}>Duration:</span>
                 <Input 
                   type="number" 
                   value={config.simulation.duration || 1.0} 
                   onChange={(e) => updateConfig('simulation.duration', parseFloat(e.target.value))}
                   placeholder="Seconds"
                   step="0.1"
                 />
               </div>
               <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
                 <span style={{ minWidth: '80px' }}>Sample Rate:</span>
                 <Input 
                   type="number" 
                   value={config.simulation.sample_rate || 44100} 
                   onChange={(e) => updateConfig('simulation.sample_rate', parseFloat(e.target.value))}
                   placeholder="Hz"
                 />
               </div>
               <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
                 <span style={{ minWidth: '80px' }}>Source Freq:</span>
                 <Input 
                   type="number" 
                   value={config.sources[0]?.signal?.f0 || 440} 
                   onChange={(e) => updateConfig('sources.0.signal.f0', parseFloat(e.target.value))}
                   placeholder="Hz"
                 />
               </div>
             </div>
           </FormGroup>
         )}

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

        {/* Audio File Upload */}
        <FormGroup style={{ marginTop: '10px' }}>
          <Label>Custom Audio Source (Optional)</Label>
          <input
            type="file"
            accept=".wav,.mp3,.flac"
            onChange={(e) => {
              const file = e.target.files?.[0];
              setCustomAudioFile(file || null);
              if (file) {
                console.log('Audio file selected:', file.name);
              }
            }}
            style={{ width: '100%', padding: '5px' }}
          />
          {customAudioFile && (
            <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
              Selected: {customAudioFile.name} ({(customAudioFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}
        </FormGroup>

        {/* Sensor Management */}
        <FormGroup style={{ marginTop: '20px' }}>
          <Label>Sensor Configuration</Label>
          <div style={{ border: '1px solid #ddd', padding: '10px', borderRadius: '4px' }}>
            {config.output.sensors.map((sensor: any, index: number) => (
              <div key={index} style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '10px' }}>
                <input
                  type="text"
                  placeholder="Sensor ID"
                  value={sensor.id || `sensor_${index + 1}`}
                  onChange={(e) => {
                    const newSensors = [...config.output.sensors];
                    newSensors[index] = { ...newSensors[index], id: e.target.value };
                    setConfig({ ...config, output: { ...config.output, sensors: newSensors } });
                  }}
                  style={{ width: '100px', padding: '3px' }}
                />
                <input
                  type="number"
                  placeholder="X"
                  value={sensor.position[0]}
                  step="0.1"
                  onChange={(e) => {
                    const newSensors = [...config.output.sensors];
                    newSensors[index].position[0] = parseFloat(e.target.value) || 0;
                    setConfig({ ...config, output: { ...config.output, sensors: newSensors } });
                  }}
                  style={{ width: '60px', padding: '3px' }}
                />
                <input
                  type="number"
                  placeholder="Y"
                  value={sensor.position[1]}
                  step="0.1"
                  onChange={(e) => {
                    const newSensors = [...config.output.sensors];
                    newSensors[index].position[1] = parseFloat(e.target.value) || 0;
                    setConfig({ ...config, output: { ...config.output, sensors: newSensors } });
                  }}
                  style={{ width: '60px', padding: '3px' }}
                />
                <input
                  type="number"
                  placeholder="Z"
                  value={sensor.position[2]}
                  step="0.1"
                  onChange={(e) => {
                    const newSensors = [...config.output.sensors];
                    newSensors[index].position[2] = parseFloat(e.target.value) || 0;
                    setConfig({ ...config, output: { ...config.output, sensors: newSensors } });
                  }}
                  style={{ width: '60px', padding: '3px' }}
                />
                <Button
                  onClick={() => {
                    const newSensors = config.output.sensors.filter((_: any, i: number) => i !== index);
                    setConfig({ ...config, output: { ...config.output, sensors: newSensors } });
                    if (selectedSensorIndex >= newSensors.length) {
                      setSelectedSensorIndex(Math.max(0, newSensors.length - 1));
                    }
                  }}
                  style={{ padding: '3px 8px', fontSize: '12px', backgroundColor: '#dc3545', color: 'white', border: 'none' }}
                >
                  Remove
                </Button>
              </div>
            ))}
            <Button
              onClick={() => {
                const newSensors = [...config.output.sensors, { id: `sensor_${config.output.sensors.length + 1}`, position: [0, 0, 0] }];
                setConfig({ ...config, output: { ...config.output, sensors: newSensors } });
              }}
              style={{ padding: '5px 10px', fontSize: '12px', backgroundColor: '#28a745', color: 'white', border: 'none' }}
            >
              + Add Sensor
            </Button>
          </div>
        </FormGroup>

        {/* Unified Simulation Button */}
        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
          <Button 
            onClick={computeTimeDomainSimulation} 
            disabled={isComputingAudio}
            style={{ 
              backgroundColor: '#17a2b8', 
              borderColor: '#17a2b8',
              color: 'white'
            }}
          >
            {isComputingAudio ? 'Computing...' : '🌊 Time Domain (Echoes)'}
          </Button>
          
          <Button 
            onClick={computeUnifiedSimulation} 
            disabled={isComputingAudio}
            style={{ backgroundColor: '#28a745', borderColor: '#28a745' }}
          >
            {isComputingAudio ? 'Computing...' : 'Run Complete Analysis'}
          </Button>
        </div>

        {/* Visualization Mode Toggle */}
        {(unifiedData || timeDomainData) && (
          <FormGroup style={{ marginTop: '20px' }}>
            <Label>Visualization Mode</Label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <Button 
                onClick={() => setVisualizationMode('frequency')}
                style={{
                  backgroundColor: visualizationMode === 'frequency' ? '#007bff' : 'transparent',
                  color: visualizationMode === 'frequency' ? 'white' : '#007bff',
                  border: '1px solid #007bff',
                  padding: '5px 10px',
                  fontSize: '12px'
                }}
              >
                Frequency Domain
              </Button>
              <Button 
                onClick={() => setVisualizationMode('time')}
                style={{
                  backgroundColor: visualizationMode === 'time' ? '#007bff' : 'transparent',
                  color: visualizationMode === 'time' ? 'white' : '#007bff',
                  border: '1px solid #007bff',
                  padding: '5px 10px',
                  fontSize: '12px'
                }}
              >
                Time Domain
              </Button>
            </div>
          </FormGroup>
        )}

        {/* Frequency Slider for Frequency Domain */}
        {unifiedData && unifiedData.frequency_data && visualizationMode === 'frequency' && availableFrequencies.length > 0 && (
          <FormGroup>
            <Label>
              Frequency: {availableFrequencies[currentFrequencyIndex]?.toFixed(1)} Hz 
              ({currentFrequencyIndex + 1}/{availableFrequencies.length})
            </Label>
            <input
              type="range"
              min={0}
              max={availableFrequencies.length - 1}
              value={currentFrequencyIndex}
              onChange={(e) => {
                const index = parseInt(e.target.value);
                updateFrequencyVisualization(
                  unifiedData,
                  index,
                  availableFrequencies,
                  setCurrentFrequencyIndex,
                  setSelectedFrequency,
                  setFieldData,
                  jobStatus?.job_id
                );
              }}
              style={{ width: '100%' }}
            />
          </FormGroup>
        )}

        {/* Time Domain Controls */}
        {unifiedData && visualizationMode === 'time' && timeDomainData && (
          <div style={{ marginTop: '20px' }}>
            <FormGroup>
              <Label>
                Time: {timeDomainData.time_field_data?.time_steps?.[currentTimeStep]?.toFixed(3)}s
                ({currentTimeStep + 1}/{timeDomainData.time_field_data?.time_steps?.length || 0})
              </Label>
              <input
                type="range"
                min={0}
                max={(timeDomainData.time_field_data?.time_steps?.length || 1) - 1}
                value={currentTimeStep}
                onChange={(e) => {
                  const step = parseInt(e.target.value);
                  setCurrentTimeStep(step);
                  console.log(`Updated time step to ${step}`);
                }}
                style={{ width: '100%' }}
              />
            </FormGroup>
            
            <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
              <Button 
                onClick={() => setIsPlayingAnimation(!isPlayingAnimation)}
                style={{
                  backgroundColor: isPlayingAnimation ? '#dc3545' : '#28a745',
                  color: 'white',
                  border: 'none',
                  padding: '5px 15px'
                }}
              >
                {isPlayingAnimation ? 'Pause' : 'Play'} Animation
              </Button>
              
              {/* Sensor Selector for Audio Playback */}
              {config.output.sensors.length > 1 && timeDomainData?.sensor_time_series && (
                <FormGroup style={{ margin: '0', minWidth: '150px' }}>
                  <Label style={{ fontSize: '12px', marginBottom: '2px' }}>Listen at Sensor:</Label>
                  <select
                    value={selectedSensorIndex}
                    onChange={(e) => setSelectedSensorIndex(parseInt(e.target.value))}
                    style={{ padding: '3px', fontSize: '12px' }}
                  >
                    {config.output.sensors.map((sensor: any, index: number) => (
                      <option key={index} value={index}>
                        {sensor.id || `Sensor ${index + 1}`}
                      </option>
                    ))}
                  </select>
                </FormGroup>
              )}
              
              {/* Audio Gain Control */}
              {timeDomainData?.sensor_time_series && (
                <FormGroup style={{ margin: '0', minWidth: '120px' }}>
                  <Label style={{ fontSize: '12px', marginBottom: '2px' }}>Audio Gain:</Label>
                  <input
                    type="range"
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    value={audioGain}
                    onChange={(e) => setAudioGain(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '10px', textAlign: 'center' }}>{audioGain}x</div>
                </FormGroup>
              )}
              
              <Button 
                onClick={playAudio}
                disabled={!timeDomainData?.sensor_time_series}
                style={{
                  backgroundColor: '#007bff',
                  color: 'white',
                  border: 'none',
                  padding: '5px 15px'
                }}
              >
                🔊 Play Audio
              </Button>
            </div>
          </div>
        )}

        {/* Legacy Frequency Selection for Visualization */}
        {results && results.frequencies && results.frequencies.length > 0 && !unifiedData && (
          <FormGroup>
            <Label>Visualization Frequency</Label>
            <Select 
              value={selectedFrequency} 
               onChange={async (e) => {
                 const freq = parseFloat(e.target.value);
                 setSelectedFrequency(freq);
                 if (jobStatus?.job_id && config.simulation.type === 'frequency_domain') {
                   await fetchFieldData(jobStatus.job_id, freq);
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

         {/* Time Domain Controls */}
         {timeDomainData && (
           <>
             <FormGroup>
               <Label>Time Domain Simulation</Label>
               <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexDirection: 'column' }}>
                 <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                   <Button 
                     onClick={() => {
                       if (!timeDomainData) {
                         computeTimeDomainSimulation();
                       } else {
                         playAudio();
                       }
                     }}
                     disabled={isComputingAudio}
                     style={{ background: timeDomainData ? '#28a745' : '#007bff' }}
                   >
                     {isComputingAudio ? '🔄 Computing...' : 
                      timeDomainData ? '🔊 Play Audio' : '🎵 Generate Time Domain'}
                   </Button>
                   {timeDomainData?.sensor_time_series && (
                     <Button 
                       onClick={playAudio}
                       style={{ background: '#17a2b8' }}
                     >
                       🔊 Play Again
                     </Button>
                   )}
                 </div>
                 <span style={{ fontSize: '12px', color: '#666' }}>
                   {timeDomainData ? 
                     `Time domain ready (${timeDomainData.duration || timeDomainData.parameters?.duration || 'N/A'}s, ${timeDomainData.num_time_steps || timeDomainData.parameters?.num_time_steps || 'N/A'} steps)` :
                     'Click to compute time-domain simulation with audio and visualization'
                   }
                 </span>
               </div>
             </FormGroup>

             {/* Time Animation Controls */}
             {timeDomainData && (
               <FormGroup>
                 <Label>Wave Animation</Label>
                 <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexDirection: 'column' }}>
                   <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                     <Button 
                       onClick={() => setIsPlayingAnimation(!isPlayingAnimation)}
                       style={{ background: isPlayingAnimation ? '#dc3545' : '#28a745' }}
                     >
                       {isPlayingAnimation ? '⏸️ Pause' : '▶️ Play Animation'}
                     </Button>
                     <span style={{ fontSize: '12px', color: '#666' }}>
                       Time: {timeDomainData.time_field_data?.time_steps?.[currentTimeStep]?.toFixed(3) || 0}s
                     </span>
                   </div>
                   <input
                     type="range"
                     min="0"
                     max={(timeDomainData.time_field_data?.num_time_steps || 1) - 1}
                     value={currentTimeStep}
                     onChange={(e) => setCurrentTimeStep(parseInt(e.target.value))}
                     style={{ width: '100%' }}
                   />
                   <span style={{ fontSize: '10px', color: '#666' }}>
                     Step {currentTimeStep} of {timeDomainData.time_field_data?.num_time_steps || 0}
                   </span>
                 </div>
               </FormGroup>
             )}
           </>
        )}
      </Sidebar>

      <MainArea>
        <Viewer>
          <Scene3D 
            config={config} 
            sources={config.sources}
            sensors={config.output.sensors}
            meshData={meshData}
            fieldData={visualizationMode === 'frequency' ? fieldData : null}
            selectedFrequency={selectedFrequency}
            timeDomainData={visualizationMode === 'time' ? timeDomainData : null}
            currentTimeStep={currentTimeStep}
            visualizationMode={visualizationMode}
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
                   height: 200,
                   margin: { t: 40, b: 40, l: 60, r: 20 }
                }}
                 style={{ width: '100%', height: '200px' }}
              />
            </div>
          )}
        </Controls>
      </MainArea>
    </Container>
  );
};

export default App;
