const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Backend communication
  backendRequest: (config) => ipcRenderer.invoke('backend-request', config),
  
  // WebSocket communication
  websocketConnect: (url) => ipcRenderer.invoke('websocket-connect', url),
  
  // Menu event listeners
  onMenuNewSimulation: (callback) => ipcRenderer.on('menu-new-simulation', callback),
  onMenuRunSimulation: (callback) => ipcRenderer.on('menu-run-simulation', callback),
  onMenuStopSimulation: (callback) => ipcRenderer.on('menu-stop-simulation', callback),
  
  // WebSocket event listeners
  onWebSocketConnected: (callback) => ipcRenderer.on('websocket-connected', callback),
  onWebSocketMessage: (callback) => ipcRenderer.on('websocket-message', callback),
  onWebSocketClosed: (callback) => ipcRenderer.on('websocket-closed', callback),
  onWebSocketError: (callback) => ipcRenderer.on('websocket-error', callback),
  
  // Remove listeners
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});
