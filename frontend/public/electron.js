const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');

let mainWindow;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'icon.png'), // Add an icon if you have one
    title: 'Acoustic Room Simulator',
    show: false // Don't show until ready
  });

  // Load the app
  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Open DevTools in development
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    require('electron').shell.openExternal(url);
    return { action: 'deny' };
  });
}

// App event handlers
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Create application menu
const template = [
  {
    label: 'File',
    submenu: [
      {
        label: 'New Simulation',
        accelerator: 'CmdOrCtrl+N',
        click: () => {
          mainWindow.webContents.send('menu-new-simulation');
        }
      },
      {
        label: 'Exit',
        accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
        click: () => {
          app.quit();
        }
      }
    ]
  },
  {
    label: 'Simulation',
    submenu: [
      {
        label: 'Run Simulation',
        accelerator: 'CmdOrCtrl+R',
        click: () => {
          mainWindow.webContents.send('menu-run-simulation');
        }
      },
      {
        label: 'Stop Simulation',
        accelerator: 'CmdOrCtrl+Shift+R',
        click: () => {
          mainWindow.webContents.send('menu-stop-simulation');
        }
      }
    ]
  },
  {
    label: 'View',
    submenu: [
      {
        label: 'Reload',
        accelerator: 'CmdOrCtrl+Shift+R',
        click: () => {
          mainWindow.reload();
        }
      },
      {
        label: 'Toggle Developer Tools',
        accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
        click: () => {
          mainWindow.webContents.toggleDevTools();
        }
      },
      {
        label: 'Toggle Fullscreen',
        accelerator: process.platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11',
        click: () => {
          mainWindow.setFullScreen(!mainWindow.isFullScreen());
        }
      }
    ]
  },
  {
    label: 'Help',
    submenu: [
      {
        label: 'About Acoustic Simulator',
        click: () => {
          const { dialog } = require('electron');
          dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'About Acoustic Room Simulator',
            message: 'Acoustic Room Simulator v1.0',
            detail: 'A finite element method-based acoustic simulation tool for room acoustics analysis.\n\nBuilt with React, Three.js, and Electron.'
          });
        }
      }
    ]
  }
];

// macOS specific menu adjustments
if (process.platform === 'darwin') {
  template.unshift({
    label: app.getName(),
    submenu: [
      { label: 'About ' + app.getName(), role: 'about' },
      { type: 'separator' },
      { label: 'Services', role: 'services' },
      { type: 'separator' },
      { label: 'Hide ' + app.getName(), accelerator: 'Command+H', role: 'hide' },
      { label: 'Hide Others', accelerator: 'Command+Shift+H', role: 'hideothers' },
      { label: 'Show All', role: 'unhide' },
      { type: 'separator' },
      { label: 'Quit', accelerator: 'Command+Q', click: () => app.quit() }
    ]
  });
}

const menu = Menu.buildFromTemplate(template);
Menu.setApplicationMenu(menu);

// IPC handlers for backend communication
ipcMain.handle('backend-request', async (event, { url, method, data }) => {
  try {
    const axios = require('axios');
    const response = await axios({
      method: method || 'GET',
      url: url,
      data: data,
      timeout: 30000
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.message,
      status: error.response?.status
    };
  }
});

// Handle WebSocket connections
ipcMain.handle('websocket-connect', (event, url) => {
  const WebSocket = require('ws');
  const ws = new WebSocket(url);
  
  ws.on('open', () => {
    event.reply('websocket-connected');
  });
  
  ws.on('message', (data) => {
    event.reply('websocket-message', data.toString());
  });
  
  ws.on('close', () => {
    event.reply('websocket-closed');
  });
  
  ws.on('error', (error) => {
    event.reply('websocket-error', error.message);
  });
  
  return ws;
});
