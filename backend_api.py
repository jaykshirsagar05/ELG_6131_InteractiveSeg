from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import datetime
from pathlib import Path
import json
import logging
from fastapi.responses import FileResponse
import uuid
from pydantic import BaseModel

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

# Define request model for retraining
class RetrainRequest(BaseModel):
    min_feedback_samples: int

@app.post("/retrain-model/")
async def trigger_model_retraining(request: RetrainRequest) -> dict:
    """
    Triggers model retraining if enough feedback samples exist for the current version.
    """
    logger.info(f"Received retraining request with min_samples: {request.min_feedback_samples}")

    try:
        current_version_str, version_info = get_current_model_version()
        current_version_data = None
        for v in version_info["versions"]:
            if v["version"] == current_version_str:
                current_version_data = v
                break

        if not current_version_data:
            logger.error(f"Could not find data for current model version {current_version_str}")
            return {"error": f"Data for current model version {current_version_str} not found."}

        feedback_count = current_version_data.get("feedback_samples", 0)
        logger.info(f"Current version {current_version_str} has {feedback_count} feedback samples.")

        if feedback_count < request.min_feedback_samples:
            message = f"Insufficient feedback samples ({feedback_count}) for version {current_version_str}. Minimum required: {request.min_feedback_samples}."
            logger.warning(message)
            return {"message": message, "status": "skipped"}

        # --- Retraining Logic Starts Here ---
        logger.info(f"Threshold met. Proceeding with retraining for version {current_version_str}.")

        # 1. Gather feedback data (CT/edited mask pairs)
        training_data_pairs = []
        feedback_dirs = [d for d in FEEDBACK_DIR.iterdir() if d.is_dir()]
        for fb_dir in feedback_dirs:
            metadata_path = fb_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                # Check if feedback corresponds to the version being retrained
                if metadata.get("model_version") == current_version_str:
                    ct_path = fb_dir / metadata.get("ct_file_relative")
                    edited_seg_path = fb_dir / metadata.get("edited_segmentation_relative")

                    if ct_path and edited_seg_path and ct_path.exists() and edited_seg_path.exists():
                        training_data_pairs.append((str(ct_path), str(edited_seg_path)))
                    else:
                         logger.warning(f"Skipping feedback {fb_dir.name}: Missing or invalid CT/edited seg file paths in metadata or files not found.")

        if not training_data_pairs:
             logger.error("No valid training pairs found despite meeting feedback count threshold.")
             return {"error": "Failed to gather valid training data."}

        logger.info(f"Gathered {len(training_data_pairs)} CT/edited mask pairs for retraining.")

        # 2. *** PLACEHOLDER: Execute the actual model training/fine-tuning ***
        #    This is where you would integrate your specific training script/library.
        #    Example:
        #    try:
        #        new_model_path = run_training_script(training_data_pairs, current_model_path)
        #        logger.info(f"Training successful. New model saved at: {new_model_path}")
        #    except Exception as train_error:
        #        logger.error(f"Model training failed: {str(train_error)}")
        #        return {"error": f"Model training process failed: {str(train_error)}"}
        logger.warning("<<<<< Placeholder: Actual model training logic goes here >>>>>")
        # Simulate success for now
        training_successful = True
        new_model_version_str = f"v{float(current_version_str.replace('v','')) + 0.1:.1f}" # Example version increment

        if not training_successful:
             return {"error": "Training process failed (Simulated)."}


        # 3. Update model version information
        logger.info(f"Training complete. Updating model version to {new_model_version_str}.")
        new_version_entry = {
            "version": new_model_version_str,
            "date_created": datetime.datetime.now().isoformat(),
            "feedback_samples": 0, # Reset feedback count for the new version
            "based_on_version": current_version_str,
            "feedback_used": len(training_data_pairs)
        }
        version_info["versions"].append(new_version_entry)
        version_info["current_version"] = new_model_version_str

        with open(MODEL_VERSION_FILE, "w") as f:
            json.dump(version_info, f, indent=2)

        logger.info(f"Successfully updated model version info. New current version: {new_model_version_str}")

        return {
            "message": f"Retraining completed successfully. New model version '{new_model_version_str}' is now active.",
            "new_version": new_model_version_str,
            "feedback_samples_used": len(training_data_pairs),
            "status": "completed"
        }
        # --- Retraining Logic Ends Here ---

    except Exception as e:
        logger.error(f"Error during retraining process: {str(e)}", exc_info=True)
        return {"error": f"Retraining process failed: {str(e)}"}

@app.post("/submit-feedback/")
async def submit_feedback(
    ct_file: str = Form(...),
    original_seg_file: str = Form(...),
    edited_seg_file: UploadFile = File(...),
    feedback_notes: str = Form(None)
) -> dict:
    """
    Submit human feedback. Copies the original CT and saves the edited mask
    into a dedicated feedback directory. Now expects ct_file, original_seg_file,
    and feedback_notes as form fields.
    """
    feedback_id = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    feedback_dir = FEEDBACK_DIR / feedback_id
    feedback_dir.mkdir(exist_ok=True)
    logger.info(f"Creating feedback directory: {feedback_dir}")

    # Convert the received strings (which are paths) to Path objects
    original_ct_path = Path(ct_file)
    original_seg_path = Path(original_seg_file) # Path to the originally generated seg

    try:
        # Define paths within the feedback directory
        ct_copy_path = feedback_dir / original_ct_path.name
        edited_seg_path = feedback_dir / "edited_segmentation.nii.gz" # Standardize edited name
        original_seg_copy_path = feedback_dir / original_seg_path.name # Optional: copy original seg too

        # 1. Save the uploaded edited segmentation mask
        logger.info(f"Saving uploaded edited segmentation to: {edited_seg_path}")
        await save_upload_file(edited_seg_file, edited_seg_path)

        # 2. Copy the original CT scan into the feedback directory
        if original_ct_path.exists():
            logger.info(f"Copying original CT from {original_ct_path} to {ct_copy_path}")
            shutil.copy2(original_ct_path, ct_copy_path)
        else:
            logger.error(f"Original CT file not found at: {original_ct_path}. Cannot copy.")
            ct_copy_path = None # Indicate copy failed

        # 3. [Optional] Copy the original segmentation into the feedback directory
        # We'll copy the original segmentation for completeness in the feedback folder
        original_seg_copy_path = None # Default to None
        if original_seg_path.exists():
            original_seg_copy_path = feedback_dir / original_seg_path.name
            logger.info(f"Copying original segmentation from {original_seg_path} to {original_seg_copy_path}")
            shutil.copy2(original_seg_path, original_seg_copy_path)
        else:
             logger.warning(f"Original segmentation file not found at: {original_seg_path}. Cannot copy.")


        # 4. Save metadata about this feedback
        metadata = {
            "ct_file_relative": ct_copy_path.name if ct_copy_path else None,
            "original_ct_path_recorded": str(original_ct_path), # Keep original path string passed from frontend
            "original_segmentation_relative": original_seg_copy_path.name if original_seg_copy_path else None,
            "original_segmentation_path_recorded": str(original_seg_path), # Keep original path string passed from frontend
            "edited_segmentation_relative": edited_seg_path.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "notes": feedback_notes,
            "model_version": get_current_model_version()[0]
        }
        metadata_path = feedback_dir / "metadata.json"
        logger.info(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 5. Update the model version info to track feedback samples
        logger.info("Updating feedback sample count for the current model version.")
        # Use a lock or ensure atomic read/write in a concurrent environment if needed
        with open(MODEL_VERSION_FILE, "r+") as f:
            version_info = json.load(f)
            current_version_str = version_info["current_version"]
            version_updated = False
            for version_entry in version_info["versions"]:
                if version_entry["version"] == current_version_str:
                    version_entry["feedback_samples"] = version_entry.get("feedback_samples", 0) + 1
                    version_updated = True
                    break
            if not version_updated:
                 logger.error(f"Could not find current version {current_version_str} in version info file to update count.")

            f.seek(0)
            json.dump(version_info, f, indent=2)
            f.truncate()

        return {
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully. Thank you!"
        }

    except Exception as e:
         logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
         # Consider adding more specific error handling if needed
         return {"error": f"Failed to process feedback: {str(e)}"}

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
