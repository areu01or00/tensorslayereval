import os
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import torch
import numpy as np
from typing import Optional, List, Any
import re # Import regex module
from pydantic import BaseModel # Added BaseModel
import time # For logging

# Assuming your existing explorers are in the same directory or accessible in PYTHONPATH
from ai_tensor_explorer import AITensorExplorer
from safetensors_explorer_cli import SafetensorsExplorer

# Configuration
# Read model path from environment variable or use a default
DEFAULT_MODEL_PATH_FALLBACK = "./Qwen_0.6B" # Fallback if MODEL_PATH env var is not set
MODEL_DIRECTORY = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH_FALLBACK)

print(f"Attempting to load model from: {MODEL_DIRECTORY}")

# Attempt to find a .safetensors file in the MODEL_DIRECTORY
model_files = list(Path(MODEL_DIRECTORY).glob('*.safetensors'))
if not model_files:
    print(f"Warning: No .safetensors files found in {MODEL_DIRECTORY}. The application might not work correctly.")
    initial_weight_file_path = ""
else:
    initial_weight_file_path = str(model_files[0])


app = FastAPI()

# Mount static files and templates
# Ensure these directories exist: ./static and ./templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)
if not templates_dir.exists():
    templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Global variable to hold the explorer instance
# This is a simple way for a single-user, local app.
# For multi-user or production, you'd need a more robust way to manage state.
try:
    # AITensorExplorer expects a list of weight files.
    # We'll use the first .safetensors file found.
    ai_explorer_instance = AITensorExplorer(weight_files=[initial_weight_file_path] if initial_weight_file_path else [])
except Exception as e:
    print(f"Error initializing AITensorExplorer: {e}")
    print("Ensure your model path is correct and .safetensors files are present.")
    # Assign a None or dummy object if initialization fails, to allow app to at least start
    # and potentially provide an error message in the UI.
    ai_explorer_instance = None 

MAX_DIM_SIZE_FOR_FULL_2D_RETURN = 128 # Max dimension size for auto-returning full 2D tensor
DEFAULT_SLICE_SIZE = 32 # Default slice size if tensor is too large and no slice params given

# Define an approximate order for common tensor types within a layer for Y-axis sorting
# Lower number means higher up (smaller Y coordinate)
TENSOR_TYPE_ORDER = {
    # Embeddings & Pre-LN
    "model.embed_tokens.weight": -100, # Special, typically first
    "model.norm.weight": -90, # Older model final norm
    "model.final_layernorm.weight": -90, # For some newer models
    "tok_embeddings.weight": -100, # Alternative embedding name
    "embed_in.weight": -100, # Another alternative

    # Layer-specific components - order them logically if possible
    "input_layernorm.weight": 0,
    "pre_attention_layernorm.weight": 0,
    "self_attn.q_proj.weight": 10,
    "self_attn.k_proj.weight": 20,
    "self_attn.v_proj.weight": 30,
    "self_attn.o_proj.weight": 40,
    "self_attn_norm.weight": 45, # Sometimes present
    "post_attention_layernorm.weight": 50,
    "pre_mlp_layernorm.weight": 50, # sometimes this name
    
    "mlp.gate_proj.weight": 60,
    "mlp.up_proj.weight": 70,
    "mlp.down_proj.weight": 80,
    "mlp_norm.weight": 85, # Sometimes present

    # Layer Biases (less common in safetensors, but if they exist)
    "self_attn.q_proj.bias": 11,
    "self_attn.k_proj.bias": 21,
    "self_attn.v_proj.bias": 31,
    "self_attn.o_proj.bias": 41,
    "mlp.gate_proj.bias": 61,
    "mlp.up_proj.bias": 71,
    "mlp.down_proj.bias": 81,

    # Final components
    "lm_head.weight": 9999, # Special, typically last
    "output.weight": 9999, # Alternative output name
    "score.weight": 9999, # Another alternative
}
# Max layer idx to handle lm_head type tensors if no layers are parsed
MAX_LAYER_FALLBACK = 1000 

# Pydantic models for tensor editing
class SliceComponent(BaseModel):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None # Optional, defaults to 1 if not specified but start/stop are

    def to_slice(self) -> slice:
        if self.start is None and self.stop is None and self.step is None:
            return slice(None) # Represents ":"
        return slice(self.start, self.stop, self.step)

class TensorEditPayload(BaseModel):
    slice_components: List[SliceComponent] # Defines the slice to edit
    new_values: Any # Could be a nested list, a flat list (if shape matches slice), or a single value for broadcasting

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    model_display_path = "N/A"
    explorer_ready = False
    if ai_explorer_instance and hasattr(ai_explorer_instance, 'weight_files') and ai_explorer_instance.weight_files:
        # ai_explorer_instance.model_path is the directory, weight_files[0] is the specific file.
        # Let's use the directory as it's more general if multiple files were ever supported by AITensorExplorer's constructor in a merged way.
        # Or, just show the first file used for initialization.
        model_display_path = ai_explorer_instance.model_path 
        if not Path(model_display_path).is_dir(): # If model_path is actually the file path from older AITensorExplorer version
             model_display_path = str(Path(ai_explorer_instance.weight_files[0]).parent)
        explorer_ready = True if ai_explorer_instance.tensors else False # Check if tensors got loaded
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "explorer_ready": explorer_ready, "model_display_path": model_display_path}
    )

@app.get("/api/tensors")
async def list_tensors_for_canvas(): # Renamed for clarity, though route is same
    if not ai_explorer_instance or not hasattr(ai_explorer_instance, 'tensors') or not ai_explorer_instance.tensors:
        print("Warning: Tensor explorer not initialized or no tensors loaded in list_tensors_for_canvas.")
        # Return empty list or error depending on how frontend should handle this
        # For now, let's allow an empty list if explorer is there but tensors aren't (e.g. model failed to load fully)
        if not ai_explorer_instance:
             raise HTTPException(status_code=503, detail="Tensor explorer not initialized.")
        return {"tensors": [], "layout_info": {"max_x": 0, "max_y_per_x": {}}}

    tensor_list_for_canvas = []
    max_parsed_layer = 0
    parsed_tensor_data = []

    for name, info in ai_explorer_instance.tensors.items():
        layer_idx = -1 # Default for non-layer tensors
        type_key_for_order = name # Start with full name for matching TENSOR_TYPE_ORDER
        specific_type = "unknown"

        # Try to parse layer index
        layer_match = re.search(r'model\.layers\.(\d+)\.(.+)', name)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            type_key_for_order = layer_match.group(2) # e.g., "self_attn.q_proj.weight"
            specific_type = type_key_for_order
            if layer_idx > max_parsed_layer:
                max_parsed_layer = layer_idx
        else:
            # Handle special tensor names that are not in typical layers
            if name == "model.embed_tokens.weight" or name.endswith("embeddings.weight") or name.endswith("embed_in.weight"):
                layer_idx = -2 # Special pre-layers group
                specific_type = "embedding"
            elif name.startswith("model.norm") or name.startswith("model.final_layernorm") :
                layer_idx = max_parsed_layer + 1 # After all parsed layers
                specific_type = "final_norm"
            elif name.startswith("lm_head") or name.startswith("output") or name.startswith("score"):
                layer_idx = max_parsed_layer + 2 # After final norm
                specific_type = "output_head"
            else: # Other non-layer tensors, group them by a common layer_idx or handle as needed
                layer_idx = -1 # General pre-computation/misc bucket
                specific_type = "misc_root"
        
        # Determine type order (Y-axis within a layer column)
        # Iterate through TENSOR_TYPE_ORDER to find the most specific match for ordering
        type_order = TENSOR_TYPE_ORDER.get(specific_type, 1000) # Default for unknown types
        best_match_len = 0
        for key, order_val in TENSOR_TYPE_ORDER.items():
            if type_key_for_order.endswith(key): # Use endswith for more specific matches like 'q_proj.weight'
                if len(key) > best_match_len: # Prefer longer (more specific) matches
                    type_order = order_val
                    best_match_len = len(key)
            elif name == key: # Direct match for global tensors like lm_head.weight
                 type_order = order_val
                 break # Exact match is best

        parsed_tensor_data.append({
            "name": name,
            "shape": info.get('shape', 'N/A'),
            "dtype": info.get('dtype', 'N/A'),
            "size_mb": info.get('size_mb', 0),
            "canvas_x": layer_idx, # Will be our X-axis (layer)
            "canvas_y_order": type_order, # Will determine Y-axis sorting within an X column
            "specific_type": specific_type # For potential coloring/grouping on frontend
        })
    
    # Adjust layer_idx for final norm and lm_head if max_parsed_layer remained 0 (e.g. only embed and lm_head)
    if max_parsed_layer == 0:
        for td in parsed_tensor_data:
            if td["canvas_x"] == 1: # was max_parsed_layer + 1
                td["canvas_x"] = MAX_LAYER_FALLBACK + 1
            elif td["canvas_x"] == 2: # was max_parsed_layer + 2
                td["canvas_x"] = MAX_LAYER_FALLBACK + 2
    else:
        # Ensure final_norm and lm_head are after the true max_parsed_layer
        for td in parsed_tensor_data:
            if td["specific_type"] == "final_norm" and td["canvas_x"] <= max_parsed_layer:
                td["canvas_x"] = max_parsed_layer + 1
            elif td["specific_type"] == "output_head" and td["canvas_x"] <= max_parsed_layer:
                 td["canvas_x"] = max_parsed_layer + 2

    # Sort tensors: primary by canvas_x, secondary by canvas_y_order, then by name for tie-breaking
    tensor_list_for_canvas = sorted(parsed_tensor_data, key=lambda t: (t["canvas_x"], t["canvas_y_order"], t["name"]))

    # For frontend, also provide info about the grid dimensions if useful
    # E.g., max_x value, and for each x, how many y items there are (or max_y)
    # This can help frontend determine overall canvas size / column widths etc.
    layout_info = {"min_x": min(t["canvas_x"] for t in tensor_list_for_canvas) if tensor_list_for_canvas else 0,
                   "max_x": max(t["canvas_x"] for t in tensor_list_for_canvas) if tensor_list_for_canvas else 0,
                   "x_coords_present": sorted(list(set(t["canvas_x"] for t in tensor_list_for_canvas)))}

    print(f"Processed {len(tensor_list_for_canvas)} tensors for canvas layout.")
    return {"tensors": tensor_list_for_canvas, "layout_info": layout_info}

@app.get("/api/tensor/{tensor_name}")
async def get_tensor_details(
    tensor_name: str,
    # Optional query parameters for slicing
    # For simplicity, supporting up to 2 dimensions for slicing via query params initially
    # These define a rectangular slice for the first two dimensions if they exist.
    dim0_start: Optional[int] = Query(None, alias="d0s"),
    dim0_end: Optional[int] = Query(None, alias="d0e"),
    dim1_start: Optional[int] = Query(None, alias="d1s"),
    dim1_end: Optional[int] = Query(None, alias="d1e"),
    # For higher dimensions, we'd need a more complex slicing mechanism or expect user to slice N-2 dims
    # For now, if >2D, these params apply to the first 2 dims, and other dims are fully included for the slice.
    # A full implementation might take a string like "0:32,0:32,:,5" for more general slicing.
):
    if not ai_explorer_instance or not hasattr(ai_explorer_instance, 'explorer'):
        print("Error: ai_explorer_instance is not properly initialized.")
        raise HTTPException(status_code=503, detail="Tensor explorer not initialized.")
    if tensor_name not in ai_explorer_instance.tensors:
        print(f"Error: Tensor '{tensor_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_name}' not found.")

    stats = None
    processed_stats = {}
    tensor_slice_data = []
    slice_applied_info = "No slicing applied or requested for visualization."
    error_log = []

    try:
        print(f"Fetching stats for tensor: {tensor_name}")
        raw_stats = ai_explorer_instance.explorer.analyze_tensor(tensor_name)
        if raw_stats:
            for key, value in raw_stats.items():
                if isinstance(value, torch.dtype):
                    processed_stats[key] = str(value)
                else:
                    processed_stats[key] = value
        stats = processed_stats # Use the processed stats
        print(f"Successfully fetched and processed stats: {stats is not None}")
    except Exception as e:
        print(f"Error in analyze_tensor for {tensor_name}: {e}")
        error_log.append(f"Failed to analyze tensor: {str(e)}")

    try:
        print(f"Loading full tensor for potential slicing: {tensor_name}")
        # Load the full tensor first using the explorer's method
        # SafetensorsExplorer.load_tensor() should return a torch tensor
        full_tensor = ai_explorer_instance.explorer.load_tensor(tensor_name)
        original_shape = list(full_tensor.shape)

        slicing_params_provided = dim0_start is not None and dim0_end is not None
        
        current_slice_obj = [slice(None)] * full_tensor.dim() # Start with full slices for all dims

        if slicing_params_provided:
            # Apply dim0 slicing if params are valid
            d0s = max(0, dim0_start if dim0_start is not None else 0)
            d0e = min(original_shape[0], dim0_end if dim0_end is not None else original_shape[0])
            if d0s < d0e:
                current_slice_obj[0] = slice(d0s, d0e)
            
            # Apply dim1 slicing if tensor has at least 2 dims and params are valid
            if full_tensor.dim() > 1 and dim1_start is not None and dim1_end is not None:
                d1s = max(0, dim1_start if dim1_start is not None else 0)
                d1e = min(original_shape[1], dim1_end if dim1_end is not None else original_shape[1])
                if d1s < d1e:
                    current_slice_obj[1] = slice(d1s, d1e)
            slice_applied_info = f"Applied slice: {tuple(current_slice_obj)}"
        elif full_tensor.dim() == 2 and (original_shape[0] > MAX_DIM_SIZE_FOR_FULL_2D_RETURN or original_shape[1] > MAX_DIM_SIZE_FOR_FULL_2D_RETURN):
            # Default slicing for large 2D tensors if no params given
            current_slice_obj[0] = slice(0, min(original_shape[0], DEFAULT_SLICE_SIZE))
            current_slice_obj[1] = slice(0, min(original_shape[1], DEFAULT_SLICE_SIZE))
            slice_applied_info = f"Default slice for large 2D: {tuple(current_slice_obj)}"
        elif full_tensor.dim() > 2: # For >2D tensors, if no slice given, take a default slice from first 2 dims
            current_slice_obj[0] = slice(0, min(original_shape[0], DEFAULT_SLICE_SIZE if original_shape[0] > DEFAULT_SLICE_SIZE else original_shape[0]))
            if full_tensor.dim() > 1:
                current_slice_obj[1] = slice(0, min(original_shape[1], DEFAULT_SLICE_SIZE if original_shape[1] > DEFAULT_SLICE_SIZE else original_shape[1]))
            # For remaining dimensions, we take only the 0-th index to make it 2D for heatmap
            for i in range(2, full_tensor.dim()):
                current_slice_obj[i] = 0 # Take the first element along higher dimensions
            slice_applied_info = f"Default 2D slice from higher-dim tensor: {tuple(current_slice_obj)}"
        
        # Perform the actual slicing
        sliced_tensor_view = full_tensor[tuple(current_slice_obj)]
        
        # Ensure the result for visualization is 2D or can be squeezed to 2D if higher dims were all size 1 after slicing
        if sliced_tensor_view.dim() > 2:
            try:
                # Attempt to squeeze out dimensions of size 1 to get to 2D if possible
                # This handles cases where higher dims were sliced to index 0
                squeezed_view = sliced_tensor_view.squeeze()
                if squeezed_view.dim() == 2:
                    sliced_tensor_view = squeezed_view
                elif squeezed_view.dim() == 1 and sliced_tensor_view.shape[0] == 1: # e.g. shape [1, N, 1, 1] -> [N]
                     # if original slice was like [0, 0:N, 0, 0], squeeze might make it 1D.
                     # Plotly heatmap typically wants 2D. We might need to reshape or handle 1D differently.
                     # For now, if it becomes 1D, we might represent it as a single row/column heatmap.
                     # If squeezed_view is 1D, convert to a 2D list with one row
                     sliced_tensor_view = squeezed_view.unsqueeze(0) # Make it [1, N]
                elif squeezed_view.dim() == 0: # Scalar result from slicing
                    sliced_tensor_view = squeezed_view.unsqueeze(0).unsqueeze(0) # Make it [[value]]
                # If still > 2D after squeeze, the slice was not effectively 2D, log and error or send as is if client can handle
                # For now, we aim for 2D for heatmap visualization
                if sliced_tensor_view.dim() > 2:
                     error_log.append(f"Slice result for visualization is still >2D: {list(sliced_tensor_view.shape)}. Heatmap may not display as expected.")
            except Exception as e_squeeze:
                error_log.append(f"Error trying to squeeze tensor slice: {str(e_squeeze)}")

        # Convert to Python list of lists (for JSON)
        # Ensure tensor is on CPU before converting to numpy then list
        
        # Handle BFloat16 conversion before numpy()
        if sliced_tensor_view.dtype == torch.bfloat16:
            print(f"Converting tensor slice from bfloat16 to float32 for serialization.")
            sliced_tensor_view = sliced_tensor_view.to(torch.float32)
            
        tensor_slice_data = sliced_tensor_view.cpu().numpy().tolist()
        print(f"Successfully sliced tensor. Slice shape: {list(sliced_tensor_view.shape)}")

    except Exception as e:
        print(f"Error in tensor loading/slicing for {tensor_name}: {e}")
        import traceback
        traceback.print_exc()
        error_log.append(f"Failed to load/slice tensor: {str(e)}")

    final_response = {
        "name": tensor_name,
        "stats": stats if stats is not None else {"error": "Statistics not available.", "shape": original_shape if 'original_shape' in locals() else [], "dtype": "unknown"},
        "tensor_slice_data": tensor_slice_data, # This is the primary data for visualization
        "slice_applied_info": slice_applied_info,
        "original_shape": original_shape if 'original_shape' in locals() else [],
        "slice_shape": list(sliced_tensor_view.shape) if 'sliced_tensor_view' in locals() and hasattr(sliced_tensor_view, 'shape') else [],
        "errors": error_log if error_log else None
    }
    print(f"Preparing to send response for {tensor_name}: Stats loaded: {stats is not None}, Slice data generated: {bool(tensor_slice_data)}")
    return final_response

@app.post("/api/tensor/{tensor_name}/edit")
async def edit_tensor_slice(tensor_name: str, payload: TensorEditPayload):
    if not ai_explorer_instance or not hasattr(ai_explorer_instance, 'explorer'):
        raise HTTPException(status_code=503, detail="Tensor explorer not initialized.")
    if tensor_name not in ai_explorer_instance.tensors:
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_name}' not found.")

    try:
        # 1. Load the original tensor
        original_tensor = ai_explorer_instance.explorer.load_tensor(tensor_name)
        if original_tensor is None:
            raise HTTPException(status_code=500, detail=f"Could not load tensor '{tensor_name}'.")
        original_dtype = original_tensor.dtype
        original_device = original_tensor.device

        # 2. Parse slice_components into a Python slice object
        slice_tuple = tuple(sc.to_slice() for sc in payload.slice_components)
        
        # Validate slice dimensionality
        if len(slice_tuple) != original_tensor.dim():
            raise HTTPException(status_code=400, detail=f"Slice dimensionality ({len(slice_tuple)}) does not match tensor dimensionality ({original_tensor.dim()}).")

        # 3. Prepare new_values
        #    - Convert to a torch.Tensor
        #    - Ensure dtype and shape compatibility with the slice
        try:
            # Attempt to get the shape of the slice view to validate new_values
            slice_view_shape = original_tensor[slice_tuple].shape
            
            # Convert new_values to tensor. Handle different input types.
            if isinstance(payload.new_values, (int, float)): # single value for broadcasting
                new_values_tensor = torch.tensor(payload.new_values, dtype=original_dtype, device=original_device)
                # Check if broadcasting is valid. If new_values_tensor is scalar, it can broadcast to any shape.
                if new_values_tensor.numel() == 1:
                    pass # Broadcasting a scalar is fine
                else: # This case should ideally not be hit if it's a single Python number
                    new_values_tensor = new_values_tensor.reshape(slice_view_shape)

            elif isinstance(payload.new_values, list):
                # Convert list to tensor, then reshape to match the slice view
                # This assumes the list contains the correct number of elements for the slice
                temp_tensor = torch.tensor(payload.new_values, dtype=original_dtype)
                new_values_tensor = temp_tensor.reshape(slice_view_shape).to(original_device)
            else:
                raise HTTPException(status_code=400, detail="new_values must be a number or a list of numbers.")

        except Exception as e:
            # This might catch errors from reshape if list has wrong number of elements, or tensor() conversion
            raise HTTPException(status_code=400, detail=f"Error processing new_values: {str(e)}. Ensure it matches the shape of the slice {list(slice_view_shape) if 'slice_view_shape' in locals() else 'unknown'} and tensor dtype {original_dtype}.")

        # 4. Perform the in-memory modification
        print(f"Attempting to edit tensor '{tensor_name}' with slice {slice_tuple} and new values of shape {new_values_tensor.shape}")
        original_tensor[slice_tuple] = new_values_tensor
        print(f"In-memory edit successful for tensor '{tensor_name}'.")

        # 5. Persist the change using TensorPatcher (Placeholder)
        # This is where you would call the TensorPatcher's method to update the underlying .safetensors file(s)
        # For example:
        # try:
        #     patch_result = ai_explorer_instance.patcher.patch_tensor(tensor_name, slice_tuple, new_values_tensor)
        #     if not patch_result.success:
        #         raise HTTPException(status_code=500, detail=f"Failed to persist tensor patch: {patch_result.message}")
        # except Exception as e:
        #     # Revert in-memory change if persistence fails? Or log and report.
        #     # For now, just raise.
        #     raise HTTPException(status_code=500, detail=f"Error during tensor persistence: {str(e)}")
        
        # For now, since actual persistence is not implemented, we'll simulate success
        # but raise NotImplementedError to indicate it's a placeholder.
        
        # TODO: Integrate with actual TensorPatcher logic here
        # raise NotImplementedError("Tensor persistence via TensorPatcher is not yet implemented.")
        
        # For the purpose of allowing the API to return something without the patcher:
        # We'll just acknowledge the in-memory change occurred.
        # In a real scenario, the response would depend on successful persistence.

        return {
            "message": f"Tensor '{tensor_name}' slice [{slice_tuple}] modified in memory. Persistence pending implementation.",
            "tensor_name": tensor_name,
            "slice_applied": str(slice_tuple),
            "new_values_shape": list(new_values_tensor.shape)
        }

    except HTTPException as he:
        raise he # Re-raise HTTPExceptions to be handled by FastAPI
    except ValueError as ve: # Catch specific errors like shape mismatches during assignment
        raise HTTPException(status_code=400, detail=f"Error applying edit to tensor '{tensor_name}': {str(ve)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while editing tensor '{tensor_name}': {str(e)}")

# TODO: Add more endpoints:
# - GET /api/tensor/{tensor_name} -> get detailed info, stats, and values for a tensor
# - POST /api/tensor/{tensor_name}/edit -> apply modifications to a tensor



if __name__ == "__main__":
    # Make sure to have `uvicorn` and `fastapi` installed:
    # pip install uvicorn fastapi python-multipart jinja2
    # Run with: python main_web.py
    # Or for development with auto-reload: uvicorn main_web:app --reload
    
    # Check if initial_weight_file_path is valid before trying to run
    if not initial_weight_file_path and (not ai_explorer_instance or not ai_explorer_instance.weight_files):
        print("CRITICAL: No model file was loaded. The AITensorExplorer could not be initialized.")
        print(f"Please ensure a .safetensors file exists in {MODEL_DIRECTORY} or update MODEL_DIRECTORY.")
        print("Application will start but tensor operations will fail.")
        # Optionally, exit here if a model is strictly required:
        # import sys
        # sys.exit("Exiting due to missing model file.")

    uvicorn.run(app, host="127.0.0.1", port=8000) 