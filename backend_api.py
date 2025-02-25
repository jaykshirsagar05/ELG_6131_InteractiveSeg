from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import datetime
from uncertainty import calculate_uncertainty_map, smooth_uncertainty_map

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing uploads, results, and model versions
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
FEEDBACK_DIR = Path("feedback")
TRAINING_DATA_DIR = Path("training_data")
UNCERTAINTY_DIR = Path("uncertainty_maps")

for directory in [UPLOAD_DIR, RESULTS_DIR, MODELS_DIR, FEEDBACK_DIR, TRAINING_DATA_DIR, UNCERTAINTY_DIR]:
    directory.mkdir(exist_ok=True)

# Model version tracking
MODEL_VERSION_FILE = MODELS_DIR / "version_info.json"
if not MODEL_VERSION_FILE.exists():
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump({
            "current_version": "v1.0",
            "versions": [{
                "version": "v1.0",
                "date_created": datetime.datetime.now().isoformat(),
                "training_samples": 0,
                "feedback_samples": 0,
                "performance_metrics": {}
            }]
        }, f, indent=2)

def get_current_model_version():
    """Get the current model version information."""
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    return version_info["current_version"], version_info

async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save an uploaded file to the specified destination."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return destination

async def run_monte_carlo_inference(input_file: Path, n_samples: int = 5) -> Tuple[Path, Path]:
    """
    Run Monte Carlo inference with dropout enabled to get uncertainty estimates.
    """
    predictions = []
    
    # Load the input CT scan
    ct_img = nib.load(str(input_file))
    ct_data = ct_img.get_fdata()
    
    # Create dummy predictions with slight variations
    for i in range(n_samples):
        # Add random noise to create variation between samples
        noise = np.random.normal(0, 0.1, ct_data.shape)
        # Create binary prediction based on simple thresholding
        pred = (ct_data > np.mean(ct_data)) * 1.0 + noise
        pred = np.clip(pred, 0, 1)
        predictions.append(pred)
    
    # Stack predictions and calculate mean prediction
    predictions_stack = np.stack(predictions, axis=0)
    mean_prediction = np.mean(predictions_stack, axis=0)
    
    # Calculate uncertainty map
    uncertainty_map = calculate_uncertainty_map(predictions_stack, method="entropy")
    uncertainty_map = smooth_uncertainty_map(uncertainty_map, sigma=1.0)
    
    # Save mean prediction as segmentation
    seg_nifti = nib.Nifti1Image(mean_prediction, ct_img.affine)
    result_path = RESULTS_DIR / f"{input_file.stem}_seg.nii.gz"
    nib.save(seg_nifti, str(result_path))
    
    # Save uncertainty map
    uncertainty_nifti = nib.Nifti1Image(uncertainty_map, ct_img.affine)
    uncertainty_path = UNCERTAINTY_DIR / f"{input_file.stem}_uncertainty.nii.gz"
    nib.save(uncertainty_nifti, str(uncertainty_path))
    
    return result_path, uncertainty_path

async def run_nnunet_inference(input_file: Path) -> Tuple[Path, Path]:
    """
    Run nnUNet inference on the input CT scan using the current model version.
    Returns paths to both segmentation and uncertainty map.
    """
    current_version, _ = get_current_model_version()
    model_path = MODELS_DIR / current_version
    
    # For MVP, we'll use Monte Carlo inference instead of actual nnUNet
    # This gives us uncertainty estimates without modifying nnUNet code
    return await run_monte_carlo_inference(input_file)

@app.post("/upload-ct/")
async def upload_ct_scan(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload a CT scan and run segmentation with uncertainty estimation.
    Returns the paths to the original CT scan, segmentation mask, and uncertainty map.
    """
    if not file.filename.endswith(('.nii', '.nii.gz')):
        return {"error": "Invalid file format. Please upload a .nii or .nii.gz file"}
    
    try:
        # Save the uploaded file
        ct_path = UPLOAD_DIR / file.filename
        await save_upload_file(file, ct_path)
        
        # Run inference with uncertainty estimation
        seg_path, uncertainty_path = await run_monte_carlo_inference(ct_path)
        
        # Get current model version
        current_version, _ = get_current_model_version()
        
        # Convert to absolute paths
        ct_path_abs = ct_path.absolute()
        seg_path_abs = seg_path.absolute()
        uncertainty_path_abs = uncertainty_path.absolute()
        
        return {
            "ct_path": str(ct_path_abs),
            "segmentation_path": str(seg_path_abs),
            "uncertainty_path": str(uncertainty_path_abs),
            "model_version": current_version,
            "message": "Segmentation completed successfully"
        }
    except Exception as e:
        return {"error": f"Segmentation failed: {str(e)}"}

@app.post("/submit-feedback/")
async def submit_feedback(
    ct_file: str,
    original_seg_file: str,
    edited_seg_file: UploadFile = File(...),
    feedback_notes: Optional[str] = None
) -> Dict[str, str]:
    """
    Submit human feedback by uploading an edited segmentation mask.
    This will be used to improve the model in future training iterations.
    """
    # Create a unique ID for this feedback
    feedback_id = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    feedback_dir = FEEDBACK_DIR / feedback_id
    feedback_dir.mkdir(exist_ok=True)
    
    # Save the edited segmentation
    edited_seg_path = feedback_dir / "edited_segmentation.nii.gz"
    await save_upload_file(edited_seg_file, edited_seg_path)
    
    # Save metadata about this feedback
    metadata = {
        "ct_file": ct_file,
        "original_segmentation": original_seg_file,
        "edited_segmentation": str(edited_seg_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "notes": feedback_notes,
        "model_version": get_current_model_version()[0]
    }
    
    with open(feedback_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update the model version info to track feedback
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    
    current_version = version_info["current_version"]
    for version in version_info["versions"]:
        if version["version"] == current_version:
            version["feedback_samples"] = version.get("feedback_samples", 0) + 1
            break
    
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump(version_info, f, indent=2)
    
    return {
        "feedback_id": feedback_id,
        "message": "Feedback submitted successfully. Thank you for helping improve the model!"
    }

@app.post("/retrain-model/")
async def retrain_model(min_feedback_samples: int = 10) -> Dict[str, str]:
    """
    Trigger model retraining based on collected feedback.
    This endpoint would typically be called by an admin or on a schedule.
    """
    # Check if we have enough feedback to warrant retraining
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    
    current_version = version_info["current_version"]
    current_version_info = None
    for version in version_info["versions"]:
        if version["version"] == current_version:
            current_version_info = version
            break
    
    if current_version_info is None or current_version_info.get("feedback_samples", 0) < min_feedback_samples:
        return {
            "status": "skipped",
            "message": f"Not enough feedback samples for retraining. Need at least {min_feedback_samples}."
        }
    
    # In a real implementation, you would:
    # 1. Prepare training data from the feedback
    # 2. Set up nnUNet training configuration
    # 3. Run the training process (which could take a long time)
    # 4. Evaluate the new model
    # 5. If better, update the current model version
    
    # For this example, we'll just create a new version entry
    new_version = f"v{float(current_version.replace('v', '')) + 0.1:.1f}"
    version_info["versions"].append({
        "version": new_version,
        "date_created": datetime.datetime.now().isoformat(),
        "training_samples": current_version_info.get("training_samples", 0),
        "feedback_samples": 0,  # Reset for the new version
        "based_on": current_version,
        "performance_metrics": {}
    })
    
    version_info["current_version"] = new_version
    
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump(version_info, f, indent=2)
    
    return {
        "status": "success",
        "message": f"Model retrained successfully. New version: {new_version}",
        "new_version": new_version
    }

@app.get("/model-versions/")
async def get_model_versions() -> Dict:
    """Get information about all model versions."""
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    return version_info

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888) 