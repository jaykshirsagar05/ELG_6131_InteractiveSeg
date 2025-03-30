import gradio as gr
import nibabel as nib
import numpy as np
import plotly.express as px
from matplotlib import cm
from PIL import Image, ImageOps
import requests
import os
from pathlib import Path
import json
import tempfile
import shutil

# Backend API URL
BACKEND_URL = "http://localhost:8888"

# --- Helper Functions ---

def load_volumes(ct_file, seg_file):
    """Load the CT and segmentation volumes from NIfTI files."""
    ct_data = nib.load(ct_file.name).get_fdata()
    seg_data = nib.load(seg_file.name).get_fdata()
    return ct_data, seg_data

def normalize_ct(slice_data):
    """Normalize a CT slice to 8-bit grayscale (0-255)."""
    if np.max(slice_data) - np.min(slice_data) == 0:
        return np.zeros(slice_data.shape, dtype=np.uint8)
    norm = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
    return norm

def colorize_segmentation(slice_data):
    """Colorize a segmentation slice using a matplotlib colormap (jet in this example)."""
    if np.max(slice_data) == 0:
        norm_data = slice_data
    else:
        norm_data = slice_data / np.max(slice_data)
    colormap = cm.get_cmap('jet')
    colored = (colormap(norm_data) * 255).astype(np.uint8)  # RGBA image
    return colored

def extract_slice(data, view, index):
    """
    Extract a 2D slice from a 3D volume.
    Assumes volume shape is (X, Y, Z):
      - axial: data[:, :, index]
      - sagittal: data[index, :, :]
      - coronal: data[:, index, :]
    """
    if view == 'axial':
        slice_data = data[:, :, index]
    elif view == 'sagittal':
        slice_data = data[index, :, :]
    elif view == 'coronal':
        slice_data = data[:, index, :]
    else:
        raise ValueError("Invalid view")
    return slice_data

def get_overlay_slice_image_from_arrays(ct_data, seg_data, view, index, alpha=0.5):
    """
    For a given view and slice index, extract the corresponding CT and segmentation slices,
    normalize and colorize them, and composite the segmentation (with transparency) over the CT image.
    """
    ct_slice = extract_slice(ct_data, view, index)
    seg_slice = extract_slice(seg_data, view, index)
    
    ct_processed = normalize_ct(ct_slice)
    seg_processed = colorize_segmentation(seg_slice)
    
    ct_img = Image.fromarray(ct_processed, mode="L").convert("RGBA")
    seg_img = Image.fromarray(seg_processed, mode="RGBA")
    seg_array = np.array(seg_img)
    seg_array[..., 3] = (seg_array[..., 3].astype(float) * alpha).astype(np.uint8)
    seg_img = Image.fromarray(seg_array, mode="RGBA")
    overlay = Image.alpha_composite(ct_img, seg_img)
    return overlay

def update_views(axial_idx, sagittal_idx, coronal_idx, alpha, volumes):
    """
    Using preloaded volumes, extract the slices for each view, overlay segmentation on CT,
    and return three Plotly figures with interactive zoom and pan.
    """
    ct_data, seg_data = volumes
    dims = ct_data.shape  # expected shape: (X, Y, Z)
    
    axial_idx = int(np.clip(axial_idx, 0, dims[2]-1))
    sagittal_idx = int(np.clip(sagittal_idx, 0, dims[0]-1))
    coronal_idx = int(np.clip(coronal_idx, 0, dims[1]-1))
    
    axial_overlay = get_overlay_slice_image_from_arrays(ct_data, seg_data, 'axial', axial_idx, alpha)
    sagittal_overlay = get_overlay_slice_image_from_arrays(ct_data, seg_data, 'sagittal', sagittal_idx, alpha)
    coronal_overlay = get_overlay_slice_image_from_arrays(ct_data, seg_data, 'coronal', coronal_idx, alpha)
    
    fig_axial = px.imshow(np.array(axial_overlay))
    fig_axial.update_layout(title=f"Axial Slice {axial_idx}", dragmode="zoom", margin=dict(l=0, r=0, t=30, b=0))
    fig_sagittal = px.imshow(np.array(sagittal_overlay))
    fig_sagittal.update_layout(title=f"Sagittal Slice {sagittal_idx}", dragmode="zoom", margin=dict(l=0, r=0, t=30, b=0))
    fig_coronal = px.imshow(np.array(coronal_overlay))
    fig_coronal.update_layout(title=f"Coronal Slice {coronal_idx}", dragmode="zoom", margin=dict(l=0, r=0, t=30, b=0))
    return fig_axial, fig_sagittal, fig_coronal

def set_volumes(ct_file, seg_file, axial_idx, sagittal_idx, coronal_idx, alpha):
    """Load the volumes from the provided files and update the views initially."""
    volumes = load_volumes(ct_file, seg_file)
    figs = update_views(axial_idx, sagittal_idx, coronal_idx, alpha, volumes)
    return volumes, *figs

def perform_segmentation(ct_file):
    """Send CT file to backend for segmentation and return the segmentation file path."""
    if ct_file is None:
        return None
    with open(ct_file.name, "rb") as f:
        files = {"file": (ct_file.name, f, "application/octet-stream")}
        response = requests.post(f"{BACKEND_URL}/upload-ct/", files=files)
    if response.status_code == 200:
        result = response.json()
        seg_path = result["segmentation_path"]
        return seg_path
    else:
        raise gr.Error(f"Segmentation failed: {response.text}")

# --- Editing Functions for the New Edit Tab ---

def load_edit_slice_from_volumes(volumes, view, slice_index, alpha=0.5):
    """
    Extract a CT slice and its corresponding segmentation slice from loaded volumes,
    and return an EditorValue dictionary for gr.ImageEditor.
    """
    ct_data, seg_data = volumes
    ct_slice = extract_slice(ct_data, view, slice_index)
    ct_img = Image.fromarray(normalize_ct(ct_slice))
    seg_slice = extract_slice(seg_data, view, slice_index)
    seg_img_color = colorize_segmentation(seg_slice)
    seg_img = Image.fromarray(seg_img_color[:, :, :3])
    editor_value = {"background": ct_img, "layers": [seg_img], "composite": seg_img}
    return editor_value

def save_edited_slice_func(edited_value, view, slice_index, edited_slices):
    """
    Save the edited slice (extracted from the editor's composite) into a dictionary keyed by (view, slice_index).
    """
    edited_slices[(view, slice_index)] = edited_value["composite"] if isinstance(edited_value, dict) else edited_value
    return f"Saved slice {slice_index} for {view} view.", edited_slices

def finalize_edited_mask(edited_slices, seg_file, view, volumes):
    """
    For the selected view, iterate over all slices in the segmentation volume.
    If an edited slice exists in edited_slices for a given slice index, use it; otherwise use the original segmentation slice.
    Reconstruct the full 3D segmentation mask and save it as a NIfTI file.
    """
    ct_data, seg_data = volumes
    dims = seg_data.shape
    if view == "axial":
        max_index = dims[2]
    elif view == "sagittal":
        max_index = dims[0]
    elif view == "coronal":
        max_index = dims[1]
    else:
        raise ValueError("Invalid view")
    final_slices = []
    for i in range(max_index):
        key = (view, i)
        if key in edited_slices:
            slice_img = np.array(edited_slices[key])
            final_slices.append(slice_img)
        else:
            slice_data = extract_slice(seg_data, view, i)
            final_slices.append(slice_data)
    if view == "axial":
        final_volume = np.stack(final_slices, axis=2)
    elif view == "sagittal":
        final_volume = np.stack(final_slices, axis=0)
    elif view == "coronal":
        final_volume = np.stack(final_slices, axis=1)
    affine = nib.load(seg_file.name).affine
    final_nifti = nib.Nifti1Image(final_volume, affine)
    out_path = Path("edited_segmentation.nii.gz")
    nib.save(final_nifti, str(out_path))
    return str(out_path)

def update_slice_slider(volumes, view):
    if volumes is None:
        return 100
    ct_data, _ = volumes
    dims = ct_data.shape
    if view == "axial":
        return dims[2] - 1
    elif view == "sagittal":
        return dims[0] - 1
    elif view == "coronal":
        return dims[1] - 1
    else:
        return 100

def submit_feedback_to_backend(ct_path, original_seg_path, edited_seg_path, notes=""):
    """Submit the edited segmentation as feedback to the backend."""
    if not all([ct_path, original_seg_path, edited_seg_path]):
        return "Error: Missing required files for feedback submission."
    
    with open(edited_seg_path, "rb") as f:
        files = {"edited_seg_file": (os.path.basename(edited_seg_path), f, "application/octet-stream")}
        data = {
            "ct_file": ct_path,
            "original_seg_file": original_seg_path,
            "feedback_notes": notes
        }
        response = requests.post(f"{BACKEND_URL}/submit-feedback/", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        return f"Feedback submitted successfully! ID: {result.get('feedback_id')}"
    else:
        return f"Error submitting feedback: {response.text}"

def get_model_versions():
    """Get the list of available model versions from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/model-versions/")
        if response.status_code == 200:
            versions = response.json()
            return versions
        else:
            return {"error": f"Failed to get model versions: {response.text}"}
    except Exception as e:
        return {"error": f"Error connecting to backend: {str(e)}"}
    
def process_segmentation_with_version(ct_file, axial_idx, sagittal_idx, coronal_idx, alpha):
    """
    Process a CT scan for segmentation.
    Returns the volumes and updated views.
    """
    if ct_file is None:
        raise gr.Error("Please upload a CT scan first")
    
    temp_dir = None
    try:
        # Send CT file to backend for segmentation
        with open(ct_file.name, "rb") as f:
            files = {"file": (os.path.basename(ct_file.name), f, "application/octet-stream")}
            response = requests.post(f"{BACKEND_URL}/upload-ct/", files=files)
        
        if response.status_code != 200:
            raise gr.Error(f"Segmentation failed: {response.text}")
        
        result = response.json()
        if "error" in result:
            raise gr.Error(f"Backend error: {result['error']}")
            
        # Create temporary file object for segmentation
        temp_dir = tempfile.mkdtemp()
        seg_file_path = os.path.join(temp_dir, "segmentation.nii")
        
        # Copy the segmentation file directly from the backend path
        shutil.copy2(result["segmentation_path"], seg_file_path)
        
        # Create a file object that Gradio can handle
        class TempFile:
            def __init__(self, path):
                self.name = path
        
        seg_file = TempFile(seg_file_path)
        
        # Load volumes
        volumes = load_volumes(ct_file, seg_file)
        
        # Update views
        figs = update_views(
            axial_idx,
            sagittal_idx,
            coronal_idx,
            alpha,
            volumes
        )
        
        model_version = result.get("model_version", "unknown")
        return (volumes,) + figs + (f"Current Model: {model_version}",)
        
    except Exception as e:
        raise gr.Error(f"Error during segmentation: {str(e)}")
    finally:
        # Clean up temporary directory after we're done with the volumes
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {str(e)}")

# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Interactive CT Scan Segmentation")
    
    # State for tracking model version
    current_model_version = gr.State("v1.0")
    
    with gr.Tabs():
        # Tab 1: Perform Segmentation
        with gr.TabItem("Perform Segmentation"):
            with gr.Row():
                ct_input_seg = gr.File(label="Upload CT Scan (.nii or .nii.gz)")
            
            with gr.Row():
                segment_btn = gr.Button("Perform Segmentation")
            
            with gr.Row():
                axial_slider_seg = gr.Slider(0, 100, step=1, label="Axial Slice", value=50)
                sagittal_slider_seg = gr.Slider(0, 100, step=1, label="Sagittal Slice", value=50)
                coronal_slider_seg = gr.Slider(0, 100, step=1, label="Coronal Slice", value=50)
            
            with gr.Row():
                alpha_slider_seg = gr.Slider(0, 1, step=0.05, label="Overlay Alpha", value=0.5)
            
            with gr.Row():
                axial_plot_seg = gr.Plot(label="Axial View")
                sagittal_plot_seg = gr.Plot(label="Sagittal View")
                coronal_plot_seg = gr.Plot(label="Coronal View")
            
            volumes_state_seg = gr.State(None)
            model_version_info = gr.Markdown("Current Model: v1.0")
            
            # Update the segmentation button click event
            segment_btn.click(
                fn=process_segmentation_with_version,
                inputs=[
                    ct_input_seg,
                    axial_slider_seg,
                    sagittal_slider_seg,
                    coronal_slider_seg,
                    alpha_slider_seg
                ],
                outputs=[
                    volumes_state_seg,
                    axial_plot_seg,
                    sagittal_plot_seg,
                    coronal_plot_seg,
                    model_version_info
                ]
            )
        
        # Tab 2: View Ground Truth and Segmentation
        with gr.TabItem("View Ground Truth and Segmentation"):
            with gr.Row():
                ct_input = gr.File(label="Upload Ground Truth CT (.nii or .nii.gz)")
                seg_input = gr.File(label="Upload Segmentation Mask (.nii or .nii.gz)")
            with gr.Row():
                axial_slider = gr.Slider(0, 100, step=1, label="Axial Slice", value=50)
                sagittal_slider = gr.Slider(0, 100, step=1, label="Sagittal Slice", value=50)
                coronal_slider = gr.Slider(0, 100, step=1, label="Coronal Slice", value=50)
            with gr.Row():
                alpha_slider = gr.Slider(0, 1, step=0.05, label="Overlay Alpha", value=0.5)
            with gr.Row():
                axial_plot = gr.Plot(label="Axial View")
                sagittal_plot = gr.Plot(label="Sagittal View")
                coronal_plot = gr.Plot(label="Coronal View")
            volumes_state = gr.State(None)
            ct_input.change(
                fn=set_volumes,
                inputs=[ct_input, seg_input, axial_slider, sagittal_slider, coronal_slider, alpha_slider],
                outputs=[volumes_state, axial_plot, sagittal_plot, coronal_plot]
            )
            seg_input.change(
                fn=set_volumes,
                inputs=[ct_input, seg_input, axial_slider, sagittal_slider, coronal_slider, alpha_slider],
                outputs=[volumes_state, axial_plot, sagittal_plot, coronal_plot]
            )
            
            def update_sliders_viewer(axial_idx, sagittal_idx, coronal_idx, alpha, volumes):
                if volumes is None:
                    return None, None, None
                return update_views(axial_idx, sagittal_idx, coronal_idx, alpha, volumes)
                
            for slider in [axial_slider, sagittal_slider, coronal_slider, alpha_slider]:
                slider.change(
                    fn=update_sliders_viewer,
                    inputs=[axial_slider, sagittal_slider, coronal_slider, alpha_slider, volumes_state],
                    outputs=[axial_plot, sagittal_plot, coronal_plot]
                )
        
        # Tab 3: Edit Segmentation and Submit Feedback
        with gr.TabItem("Edit & Submit Feedback"):
            gr.Markdown("### Edit the Segmentation Mask and Submit Feedback to Improve the Model")
            
            # Hidden states for editing
            edit_volumes_state = gr.State(None)
            edited_slices_state = gr.State({})
            ct_path_state = gr.State(None)
            original_seg_path_state = gr.State(None)
            
            # Two-column layout: left column for the editing window; right column for controls.
            with gr.Row():
                with gr.Column(scale=1):
                    edit_image_editor = gr.ImageEditor(
                        label="Edit Slice",
                        type="pil",
                        interactive=True,
                        height=800,
                        brush=gr.Brush(default_size="auto", colors=["#FF0000", "#00FF00", "#0000FF"], default_color="#FF0000", color_mode="fixed")
                    )
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("#### 1. Load Data")
                        edit_ct_input = gr.File(label="Upload CT Scan (.nii or .nii.gz)", file_count="single")
                        edit_seg_input = gr.File(label="Upload Segmentation Mask (.nii or .nii.gz)", file_count="single")
                        load_volumes_btn = gr.Button("Load Volumes for Editing")
                    
                    with gr.Group():
                        gr.Markdown("#### 2. Edit Segmentation")
                        view_selector = gr.Radio(choices=["axial", "sagittal", "coronal"], label="Select View", value="axial")
                        slice_slider = gr.Slider(0, 100, step=1, label="Slice Index", value=50)
                        load_slice_btn = gr.Button("Load Slice for Editing")
                        save_slice_btn = gr.Button("Save Edited Slice")
                        slice_save_status = gr.Textbox(label="Slice Save Status", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("#### 3. Finalize and Submit Feedback")
                        finalize_btn = gr.Button("Finalize Edited Mask")
                        final_status = gr.Textbox(label="Finalized Mask Status", interactive=False)
                        feedback_notes = gr.Textbox(label="Feedback Notes (optional)", placeholder="Describe what you changed and why...")
                        submit_feedback_btn = gr.Button("Submit Feedback to Improve Model", variant="primary")
                        feedback_status = gr.Textbox(label="Feedback Submission Status", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("#### Model Information")
                        refresh_model_info_btn = gr.Button("Refresh Model Information")
                        model_info_display = gr.JSON(label="Model Version Information")
            
            # Load volumes event
            def load_volumes_and_store_paths(ct, seg):
                """Load volumes and store the file paths for later use."""
                volumes = load_volumes(ct, seg)
                return volumes, ct.name, seg.name
            
            load_volumes_btn.click(
                fn=load_volumes_and_store_paths,
                inputs=[edit_ct_input, edit_seg_input],
                outputs=[edit_volumes_state, ct_path_state, original_seg_path_state]
            )
            
            # View selector event
            view_selector.change(
                fn=update_slice_slider,
                inputs=[edit_volumes_state, view_selector],
                outputs=slice_slider
            )
            
            # Load slice event
            load_slice_btn.click(
                fn=load_edit_slice_from_volumes,
                inputs=[edit_volumes_state, view_selector, slice_slider],
                outputs=edit_image_editor
            )
            
            # Save slice event
            save_slice_btn.click(
                fn=save_edited_slice_func,
                inputs=[edit_image_editor, view_selector, slice_slider, edited_slices_state],
                outputs=[slice_save_status, edited_slices_state]
            )
            
            # Finalize mask event
            finalize_btn.click(
                fn=finalize_edited_mask,
                inputs=[edited_slices_state, edit_seg_input, view_selector, edit_volumes_state],
                outputs=final_status
            )
            
            # Submit feedback event
            def submit_feedback_wrapper(ct_path, original_seg_path, final_status, notes):
                """Wrapper to extract the edited segmentation path from the final status message."""
                if "edited_segmentation.nii.gz" not in final_status:
                    return "Please finalize the edited mask first."
                
                edited_seg_path = "edited_segmentation.nii.gz"  # This should match the path in finalize_edited_mask
                return submit_feedback_to_backend(ct_path, original_seg_path, edited_seg_path, notes)
            
            submit_feedback_btn.click(
                fn=submit_feedback_wrapper,
                inputs=[ct_path_state, original_seg_path_state, final_status, feedback_notes],
                outputs=feedback_status
            )
            
            # Refresh model info event
            refresh_model_info_btn.click(
                fn=get_model_versions,
                inputs=[],
                outputs=model_info_display
            )
        
        # Tab 4: Model Training Dashboard (Admin)
        with gr.TabItem("Model Training (Admin)"):
            gr.Markdown("### Model Training Dashboard")
            
            with gr.Row():
                with gr.Column():
                    min_feedback_samples = gr.Slider(1, 50, step=1, value=10, label="Minimum Feedback Samples for Retraining")
                    retrain_btn = gr.Button("Trigger Model Retraining", variant="primary")
                    retrain_status = gr.Textbox(label="Retraining Status", interactive=False)
                
                with gr.Column():
                    admin_refresh_btn = gr.Button("Refresh Model Information")
                    admin_model_info = gr.JSON(label="Model Version Information")
                    
                    gr.Markdown("### Feedback Statistics")
                    feedback_stats = gr.DataFrame(
                        headers=["Version", "Feedback Samples", "Date Created"],
                        datatype=["str", "number", "str"],
                        label="Feedback by Model Version"
                    )
            
            # Retrain model event
            def trigger_retraining(min_samples):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/retrain-model/",
                        json={"min_feedback_samples": min_samples}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("message", "Retraining completed")
                    else:
                        return f"Error: {response.text}"
                except Exception as e:
                    return f"Error connecting to backend: {str(e)}"
            
            retrain_btn.click(
                fn=trigger_retraining,
                inputs=[min_feedback_samples],
                outputs=retrain_status
            )
            
            # Refresh model info and stats
            def get_model_info_and_stats():
                try:
                    response = requests.get(f"{BACKEND_URL}/model-versions/")
                    if response.status_code == 200:
                        versions = response.json()
                        
                        # Extract stats for the DataFrame
                        stats_data = []
                        for version in versions.get("versions", []):
                            stats_data.append([
                                version.get("version", "unknown"),
                                version.get("feedback_samples", 0),
                                version.get("date_created", "unknown")
                            ])
                        
                        return versions, stats_data
                    else:
                        return {"error": response.text}, []
                except Exception as e:
                    return {"error": str(e)}, []
            
            def update_model_info_and_stats():
                info, stats = get_model_info_and_stats()
                return info, stats
            
            admin_refresh_btn.click(
                fn=update_model_info_and_stats,
                inputs=[],
                outputs=[admin_model_info, feedback_stats]
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Specify port explicitly
        share=False            # Don't create a public URL
    )
