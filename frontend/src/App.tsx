import React from 'react';
import { GradioViewer } from './components/GradioViewer';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Interactive CT Scan Segmentation</h1>
      </header>
      <main className="App-main">
        <GradioViewer />
      </main>
    </div>
  );
}

export default App;
