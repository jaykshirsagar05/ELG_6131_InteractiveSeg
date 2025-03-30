from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import datetime
from pathlib import Path
import json
import logging
from fastapi.responses import FileResponse

from totalsegmentator.python_api import totalsegmentator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path of the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Directories for storing uploads, feedback, segmentations, and models
UPLOAD_DIR = SCRIPT_DIR / "uploads"
FEEDBACK_DIR = SCRIPT_DIR / "feedback"
SEGMENTATION_DIR = SCRIPT_DIR / "segmentations"
MODELS_DIR = SCRIPT_DIR / "models"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, FEEDBACK_DIR, SEGMENTATION_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)
    logger.info(f"Created/verified directory: {directory}")

# Model version tracking
MODEL_VERSION_FILE = MODELS_DIR / "version_info.json"
if not MODEL_VERSION_FILE.exists():
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump({
            "current_version": "v1.0",
            "versions": [{
                "version": "v1.0",
                "date_created": datetime.datetime.now().isoformat(),
                "feedback_samples": 0,
            }]
        }, f, indent=2)
    logger.info(f"Created model version file: {MODEL_VERSION_FILE}")

def get_current_model_version():
    """Get the current model version information."""
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    return version_info["current_version"], version_info

async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save an uploaded file to the specified destination."""
    try:
        logger.info(f"Saving uploaded file to: {destination}")
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.info(f"Successfully saved file to: {destination}")
        # Verify file exists and has content
        if destination.exists():
            size = destination.stat().st_size
            logger.info(f"File size: {size} bytes")
        else:
            logger.error(f"File was not created at: {destination}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise
    finally:
        upload_file.file.close()
    return destination

@app.post("/upload-ct/")
async def upload_ct_scan(file: UploadFile = File(...)) -> dict:
    """
    Upload a CT scan, perform segmentation using totalsegmentator's fast model,
    and return paths to the CT and segmentation files.
    """
    logger.info(f"Received file upload: {file.filename}")
    
    if not file.filename.endswith(('.nii', '.nii.gz')):
        return {"error": "Invalid file format. Please upload a .nii or .nii.gz file"}
    
    try:
        # Save the uploaded CT file
        ct_path = UPLOAD_DIR / file.filename
        logger.info(f"Attempting to save CT file to: {ct_path}")
        await save_upload_file(file, ct_path)
        
        # Determine segmentation output file name
        if file.filename.endswith('.nii.gz'):
            seg_filename = file.filename.replace('.nii.gz', '_seg.nii.gz')
        else:
            seg_filename = file.filename.replace('.nii', '_seg.nii')
        seg_path = SEGMENTATION_DIR / seg_filename
        logger.info(f"Segmentation will be saved to: {seg_path}")
        
        # Perform segmentation using totalsegmentator's fast model
        logger.info("Starting segmentation...")
        totalsegmentator(str(ct_path), str(seg_path), fast=True, ml=True)
        logger.info("Segmentation completed")
        
        # Get current model version
        current_version, _ = get_current_model_version()
        
        return {
            "ct_path": str(ct_path),
            "segmentation_path": str(seg_path),
            "model_version": current_version,
            "message": "Segmentation completed successfully"
        }
    except Exception as e:
        logger.error(f"Error in upload_ct_scan: {str(e)}")
        return {"error": f"File upload and segmentation failed: {str(e)}"}

@app.post("/submit-feedback/")
async def submit_feedback(
    ct_file: str,
    original_seg_file: str,
    edited_seg_file: UploadFile = File(...),
    feedback_notes: str = None
) -> dict:
    """
    Submit human feedback by uploading an edited segmentation mask.
    """
    feedback_id = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    feedback_dir = FEEDBACK_DIR / feedback_id
    feedback_dir.mkdir(exist_ok=True)
    
    # Save the edited segmentation file
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
    
    # Update the model version info to track feedback samples
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    
    current_version = version_info["current_version"]
    for version in version_info["versions"]:
        if version["version"] == current_version:
            version["feedback_samples"] += 1
    
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump(version_info, f, indent=2)
    
    return {
        "feedback_id": feedback_id,
        "message": "Feedback submitted successfully. Thank you!"
    }

@app.get("/model-versions/")
async def get_model_versions() -> dict:
    """Get information about all model versions."""
    with open(MODEL_VERSION_FILE, "r") as f:
        version_info = json.load(f)
    return version_info

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/download-segmentation/{file_path:path}")
async def download_segmentation(file_path: str):
    """
    Download a segmentation file from the server.
    """
    logger.info(f"Requested segmentation download: {file_path}")
    try:
        # Convert the file path to a Path object
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Segmentation file not found: {file_path}")
            return {"error": "Segmentation file not found"}
            
        # Return the file as a response
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading segmentation: {str(e)}")
        return {"error": f"Failed to download segmentation: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
