import React, { useState } from 'react';
import './GradioViewer.css';

interface GradioViewerProps {
  gradioUrl?: string;
}

export const GradioViewer: React.FC<GradioViewerProps> = ({ 
  gradioUrl = 'http://localhost:7860'  // Default Gradio URL
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const handleError = () => {
    setError("Failed to load Gradio interface. Please ensure the Gradio server is running at " + gradioUrl);
    setIsLoading(false);
  };

  return (
    <div className="gradio-container">
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner">Loading Gradio interface...</div>
        </div>
      )}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      <iframe
        src={gradioUrl}
        className="gradio-iframe"
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
        onLoad={() => setIsLoading(false)}
        onError={handleError}
        title="Gradio Interface"
      />
    </div>
  );
}; 