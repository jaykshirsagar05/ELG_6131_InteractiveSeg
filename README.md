# Interactive 3D CT Scan Segmentation with Human-in-the-Loop Learning

This project implements an interactive 3D CT scan segmentation system with human-in-the-loop learning. The system allows users to:

1. Upload CT scans for automatic segmentation
2. View and compare ground truth with segmentation results
3. Edit segmentation masks and submit feedback
4. Improve the model over time through human feedback

## Architecture

The system consists of two main components:

1. **FastAPI Backend**: Handles CT scan processing, segmentation using nnUNet, and model retraining
2. **Gradio Frontend**: Provides an interactive interface for viewing, editing, and providing feedback on segmentations

## Features

- **Automatic Segmentation**: Upload CT scans and get automatic segmentation results
- **Interactive Visualization**: View CT scans and segmentation masks in axial, sagittal, and coronal planes
- **Segmentation Editing**: Edit segmentation masks directly in the browser
- **Feedback Submission**: Submit edited masks to improve the model
- **Model Versioning**: Track model versions and improvements over time
- **Admin Dashboard**: Monitor feedback and trigger model retraining

## Setup and Installation

### Prerequisites

- Python 3.8+
- nnUNet (for segmentation)
- CUDA-capable GPU (recommended for faster inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/interactive-segmentation.git
cd interactive-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up nnUNet (follow the [official nnUNet installation guide](https://github.com/MIC-DKFZ/nnUNet))

### Running the Application

1. Start the backend server:
```bash
python backend_api.py
```

2. Start the Gradio interface:
```bash
python gradio_viewer.py
```

3. Access the interface in your browser at `http://localhost:7860`

## Usage Guide

### Performing Segmentation

1. Go to the "Perform Segmentation" tab
2. Upload a CT scan (.nii or .nii.gz format)
3. Click "Perform Segmentation"
4. View the results in the three orthogonal planes

### Viewing Ground Truth and Segmentation

1. Go to the "View Ground Truth and Segmentation" tab
2. Upload both a CT scan and a segmentation mask
3. Use the sliders to navigate through the volume

### Editing and Submitting Feedback

1. Go to the "Edit & Submit Feedback" tab
2. Upload a CT scan and its segmentation mask
3. Select a view (axial, sagittal, or coronal) and slice
4. Edit the segmentation using the drawing tools
5. Save the edited slice
6. Repeat for other slices as needed
7. Click "Finalize Edited Mask" to create a new segmentation file
8. Add optional notes and click "Submit Feedback" to send to the backend

### Model Training (Admin)

1. Go to the "Model Training (Admin)" tab
2. View feedback statistics and model version information
3. Set the minimum number of feedback samples required for retraining
4. Click "Trigger Model Retraining" to start the retraining process

## Implementation Details

### Backend

- FastAPI for API endpoints
- nnUNet for segmentation
- Model versioning and feedback tracking
- Automatic retraining based on feedback

### Frontend

- Gradio for interactive UI
- Plotly for interactive visualization
- Image editing capabilities
- Real-time feedback

## Future Improvements

- Active learning strategies to prioritize cases for human review
- Uncertainty estimation to highlight areas that need human attention
- Performance metrics tracking across model versions
- Multi-user support with authentication
- Distributed training for faster model updates

## License

[MIT License](LICENSE)

## Acknowledgments

- nnUNet for the segmentation framework
- Gradio for the interactive interface
- FastAPI for the backend framework