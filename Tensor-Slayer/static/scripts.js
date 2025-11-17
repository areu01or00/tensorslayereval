document.addEventListener('DOMContentLoaded', () => {
    // Tab switching logic for the sidebar
    const tabButtons = document.querySelectorAll('.toolkit-sidebar .tab-button');
    const tabContents = document.querySelectorAll('.toolkit-sidebar .sidebar-tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Deactivate all tabs and hide all content
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Activate the clicked tab and show its content
            button.classList.add('active');
            const targetTabContentId = button.dataset.tab;
            const targetContent = document.getElementById(targetTabContentId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    const tensorDetailsDiv = document.getElementById('tensor-details-content'); // Still in Focus tab
    
    // Slice control inputs (still in Focus tab)
    const d0sInput = document.getElementById('d0s_input');
    const d0eInput = document.getElementById('d0e_input');
    const d1sInput = document.getElementById('d1s_input');
    const d1eInput = document.getElementById('d1e_input');
    const applySliceBtn = document.getElementById('apply-slice-btn');
    
    let currentSelectedTensorName = null; 
    let allTensors = []; 
    let currentTensorSliceParams = { d0s: 0, d0e: 32, d1s: 0, d1e: 32 }; 

    // updateSliceParams and syncSliceInputsToParams are still relevant for the Focus tab
    function updateSliceParams() {
        if (!d0sInput || !d0eInput || !d1sInput || !d1eInput) return; // Guard if elements not found
        currentTensorSliceParams.d0s = parseInt(d0sInput.value) || 0;
        currentTensorSliceParams.d0e = parseInt(d0eInput.value) || 32;
        currentTensorSliceParams.d1s = parseInt(d1sInput.value) || 0;
        currentTensorSliceParams.d1e = parseInt(d1eInput.value) || 32;

        if (currentTensorSliceParams.d0e <= currentTensorSliceParams.d0s) {
            currentTensorSliceParams.d0e = currentTensorSliceParams.d0s + 1;
            d0eInput.value = currentTensorSliceParams.d0e;
        }
        if (currentTensorSliceParams.d1e <= currentTensorSliceParams.d1s) {
            currentTensorSliceParams.d1e = currentTensorSliceParams.d1s + 1;
            d1eInput.value = currentTensorSliceParams.d1e;
        }
    }

    function syncSliceInputsToParams(){
        if(d0sInput) d0sInput.value = currentTensorSliceParams.d0s;
        if(d0eInput) d0eInput.value = currentTensorSliceParams.d0e;
        if(d1sInput) d1sInput.value = currentTensorSliceParams.d1s;
        if(d1eInput) d1eInput.value = currentTensorSliceParams.d1e;
    }

    if (applySliceBtn) {
        applySliceBtn.addEventListener('click', () => {
            if (currentSelectedTensorName) {
                updateSliceParams(); 
                fetchTensorDetails(currentSelectedTensorName); 
            }
        });
    }

    async function fetchTensors() {
        console.log("Fetching tensors for canvas layout...");
        const canvasContainer = document.getElementById('tensor-canvas-container');
        if (!canvasContainer) {
            console.error("Tensor canvas container not found!");
            return;
        }
        canvasContainer.innerHTML = '<p style="text-align:center;">Loading tensor canvas...</p>'; // Update loading message

        try {
            const response = await fetch('/api/tensors');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); // Expects { tensors: [], layout_info: {} }
            allTensors = data.tensors; // Store for later use (e.g., getting full details)
            
            if (allTensors && allTensors.length > 0) {
                renderTensorCanvas(allTensors, data.layout_info, canvasContainer);
            } else {
                canvasContainer.innerHTML = '<p style="text-align:center;">No tensors found or model not loaded.</p>';
            }
        } catch (error) {
            canvasContainer.innerHTML = `<p style="text-align:center; color:red;">Error loading tensors: ${error.message}</p>`;
            console.error("Fetch error for /api/tensors:", error);
        }
    }
    
    function renderTensorCanvas(tensors, layoutInfo, container) {
        container.innerHTML = ''; // Clear previous content (e.g., loading message)
        container.style.display = 'grid'; // Use CSS Grid for layout
        
        // Determine grid dimensions from layoutInfo if useful, or adapt dynamically
        // For simplicity, we'll create columns based on unique x_coords_present
        // And rows will stack based on y_order within those columns.

        const xCoords = layoutInfo.x_coords_present || [];
        const minX = Math.min(...xCoords, 0); // Ensure minX is at least 0 or less if negative coords exist
        const maxX = Math.max(...xCoords, 0);

        // Normalize xCoords to be 0-indexed for grid-column-start
        // And create a map from original x_coord to grid column index
        const xCoordToGridColumn = {};
        const sortedUniqueXCoords = [...new Set(tensors.map(t => t.canvas_x))].sort((a, b) => a - b);
        
        sortedUniqueXCoords.forEach((xc, index) => {
            xCoordToGridColumn[xc] = index + 1; // CSS grid lines are 1-indexed
        });
        
        container.style.gridTemplateColumns = `repeat(${sortedUniqueXCoords.length || 1}, auto)`;
        // Max Y elements in any column - for now, just let grid auto-place rows
        
        tensors.forEach(tensor => {
            const tensorDiv = document.createElement('div');
            tensorDiv.classList.add('tensor-thumbnail');
            tensorDiv.style.gridColumnStart = xCoordToGridColumn[tensor.canvas_x];
            // Y-order is handled by the sort order from backend, items will fill grid cells in order.
            // For more precise Y placement within a column, one might group by X then sort by Y,
            // and then assign grid-row-start if needed, or use flexbox within each column cell.
            // For now, relying on the overall sort order.

            // Basic tensor info for display
            let shortName = tensor.name.length > 30 ? `...${tensor.name.slice(-27)}` : tensor.name;
            const shapeStr = tensor.shape && tensor.shape.length > 0 ? tensor.shape.join('x') : 'N/A';
            tensorDiv.innerHTML = `
                <div class="tensor-name">${shortName}</div>
                <div class="tensor-shape">${shapeStr}</div>
                <div class="tensor-dtype">${tensor.dtype || 'N/A'}</div>
                <div class="tensor-size">${tensor.size_mb ? tensor.size_mb.toFixed(2) + 'MB' : ''}</div>
                <div class="tensor-debug-xy">X:${tensor.canvas_x}, Y-Ord:${tensor.canvas_y_order}</div>
            `;
            // Add specific_type as a data attribute for potential styling
            tensorDiv.dataset.specificType = tensor.specific_type;
            tensorDiv.title = tensor.name; // Full name on hover

            tensorDiv.addEventListener('click', () => {
                currentSelectedTensorName = tensor.name;
                syncSliceInputsToParams(); // Reset slice inputs to defaults or last used for this tensor if stored
                fetchTensorDetails(tensor.name); // Fetch details for the Focus tab
                
                // Highlight selected tensor (optional)
                document.querySelectorAll('.tensor-thumbnail.selected').forEach(el => el.classList.remove('selected'));
                tensorDiv.classList.add('selected');
            });
            container.appendChild(tensorDiv);
        });

        if (tensors.length === 0) {
            container.innerHTML = '<p style="text-align:center;">No tensors to display. Model might be empty or filter too restrictive.</p>';
        }
    }

    // renderTensorList is now obsolete due to canvas layout
    /*
    function renderTensorList(tensorsToRender) { ... old code ... }
    */

    // tensorSearchInput is also obsolete/needs rethinking for canvas filter
    /*
    const tensorSearchInput = document.getElementById('tensor-search'); // This ID is gone
    if (tensorSearchInput) {
        tensorSearchInput.addEventListener('input', (event) => { ... });
    }
    */

    async function fetchTensorDetails(tensorName) {
        if (!tensorDetailsDiv) return; // Ensure the target div exists
        tensorDetailsDiv.innerHTML = `<p>Loading details for ${tensorName}...</p>`;
        // Ensure slice params are up-to-date before fetching (from Focus tab inputs)
        // This part is okay as slice controls are in the focus tab.
        updateSliceParams(); 
        
        const queryParams = new URLSearchParams({
            d0s: currentTensorSliceParams.d0s,
            d0e: currentTensorSliceParams.d0e,
            d1s: currentTensorSliceParams.d1s,
            d1e: currentTensorSliceParams.d1e
        }).toString();

        try {
            const response = await fetch(`/api/tensor/${encodeURIComponent(tensorName)}?${queryParams}`);
            if (!response.ok) {
                const errorData = await response.json();
                tensorDetailsDiv.innerHTML = `<p>Error loading tensor details for ${tensorName}: ${errorData.detail || 'Unknown error'}</p>`;
                if(errorData.errors) {
                    tensorDetailsDiv.innerHTML += `<p>Server errors: ${errorData.errors.join('; ')}</p>`;
                }
                console.error("Fetch details error data:", errorData);
                return; 
            }
            const details = await response.json();
            renderTensorDetails(details); // This function will also need updates to target new structure
            
            // Heatmap rendering logic will go into renderTensorDetails or be called from there,
            // targeting a div within the tensorDetailsDiv (Focus View tab)
            const heatmapContainerId = 'tensor-visualization'; // This div is now part of tensor-details-content
            const heatmapDiv = document.getElementById(heatmapContainerId);

            if (details.tensor_slice_data && details.tensor_slice_data.length > 0) {
                if (heatmapDiv) renderHeatmap(details.tensor_slice_data, heatmapContainerId, details.name, details.slice_applied_info);
                else console.error('Heatmap container not found for focus view');
            } else {
                if(heatmapDiv) heatmapDiv.innerHTML = '<p>No slice data available for heatmap visualization.</p>';
            }

        } catch (error) {
            tensorDetailsDiv.innerHTML = `<p>Error loading tensor details for ${tensorName}: ${error.message}</p>`;
            console.error("Fetch details error:", error);
        }
    }

    // renderTensorDetails needs to be adapted for the new HTML structure within the Focus tab
    function renderTensorDetails(details) {
        if (!tensorDetailsDiv) return;
        // Find the basic info from allTensors if it exists (needed if backend doesn't return all fields)
        const tensorBasicInfo = allTensors.find(t => t.name === details.name) || {}; 
        
        let html = `<h3>${details.name}</h3>`;

        if (details.errors && details.errors.length > 0) {
            html += `<p style="color: red;"><strong>Errors:</strong> ${details.errors.join('; ')}</p>`;
        }

        html += '<table>';
        const originalShape = (details.original_shape && details.original_shape.length > 0) ? details.original_shape : (tensorBasicInfo.shape || []);
        html += `<tr><th>Original Shape</th><td>${originalShape.length > 0 ? originalShape.join(' x ') : 'N/A'}</td></tr>`;
        html += `<tr><th>Data Type</th><td>${(details.stats && details.stats.dtype) ? details.stats.dtype : (tensorBasicInfo.dtype || 'N/A')}</td></tr>`;
        // size_mb might not be in details if not fetched with full tensor list yet, or if allTensors is not populated correctly
        html += `<tr><th>Size (MB)</th><td>${tensorBasicInfo.size_mb ? tensorBasicInfo.size_mb.toFixed(2) : (details.stats && details.stats.size_mb ? details.stats.size_mb.toFixed(2) : 'N/A')}</td></tr>`;
        
        if (details.stats && !details.stats.error) {
            html += `<tr><th>Min Value</th><td>${details.stats.min !== undefined ? details.stats.min.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Max Value</th><td>${details.stats.max !== undefined ? details.stats.max.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Mean</th><td>${details.stats.mean !== undefined ? details.stats.mean.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Std Dev</th><td>${details.stats.std !== undefined ? details.stats.std.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Zeros %</th><td>${details.stats.zeros_percent !== undefined ? details.stats.zeros_percent.toFixed(2) + '%': 'N/A'}</td></tr>`;
        } else if (details.stats && details.stats.error) {
            html += `<tr><th>Statistics</th><td>Error loading stats: ${details.stats.error}</td></tr>`;
        }
        html += '</table>';

        html += `<h4>Slice Details:</h4>`;
        html += `<p>Slice Applied: ${details.slice_applied_info || 'N/A'}</p>`;
        html += `<p>Slice Shape: ${details.slice_shape && details.slice_shape.length > 0 ? details.slice_shape.join(' x ') : 'N/A'}</p>`;
        
        if (details.tensor_slice_data && details.tensor_slice_data.length > 0 && details.slice_shape && details.slice_shape.reduce((a,b) => a*b, 1) < 500) {
            html += '<h4>Raw Slice Values (if small):</h4>';
            let formatted_values = "";
            if (Array.isArray(details.tensor_slice_data[0])) { 
                formatted_values = details.tensor_slice_data.map(row => 
                    '[' + row.map(val => (typeof val === 'number' ? val.toFixed(4) : val)).join(', ') + ']'
                ).join('\n');
            } else { 
                formatted_values = '[' + details.tensor_slice_data.map(val => (typeof val === 'number' ? val.toFixed(4) : val)).join(', ') + ']';
            }
            html += `<div class="value-view">${formatted_values}</div>`;
        }
        
        // The div for heatmap is now part of the HTML structure created by this function
        html += '<div id="tensor-visualization" style="margin-top: 20px; min-height: 300px; border: 1px dashed #ccc; padding:10px;"></div>';
        
        // Placeholder for editing controls in Focus tab (can be expanded later)
        // html += '<div id="tensor-editing-focus" style="margin-top: 20px;">';
        // html += '<h4>Edit Tensor (Focus View)</h4>';
        // html += '<p>Basic editing for focused slice will go here.</p>';
        // html += '</div>';

        tensorDetailsDiv.innerHTML = html;
    }

    // Initial fetch (will now attempt to render canvas)
    fetchTensors(); 

    // renderHeatmap function remains largely the same, as it targets an element by ID.
    function renderHeatmap(data, elementId, tensorName, sliceInfo) {
        const vizDiv = document.getElementById(elementId);
        if (!vizDiv) {
            console.error(`Heatmap target div '${elementId}' not found.`);
            return;
        }

        let z_data = data;
        if (!Array.isArray(data) || (data.length > 0 && !Array.isArray(data[0]))) {
            z_data = [data]; 
        }
        
        if (z_data.length > 0 && Array.isArray(z_data[0])){
            const firstRowLength = z_data[0].length;
            for(let i = 1; i < z_data.length; i++) {
                if (z_data[i].length !== firstRowLength) {
                    if (z_data[i].length > firstRowLength) z_data[i] = z_data[i].slice(0, firstRowLength);
                    else while(z_data[i].length < firstRowLength) z_data[i].push(null); 
                }
            }
        }

        const trace = {
            z: z_data,
            type: 'heatmap',
            colorscale: 'Viridis', 
            reversescale: false,
            hoverongaps: false,
            colorbar: {title: 'Value', titleside: 'right'}
        };

        const layout = {
            title: `Heatmap: ${tensorName} (${sliceInfo || 'Full View'})`,
            xaxis: { title: 'Dim 1 Index' },
            yaxis: { title: 'Dim 0 Index', autorange: 'reversed' }, 
            width: vizDiv.clientWidth > 50 ? vizDiv.clientWidth - 20 : Math.min(500, window.innerWidth * 0.3), // Responsive width, ensure it fits sidebar
            height: (z_data.length * 15 > 250 ? z_data.length * 15 : 250) + 100, // Dynamic height, more compact
            margin: { l: 50, r: 20, b: 50, t: 50, pad: 4 }
        };
        
        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot(elementId, [trace], layout, {responsive: true});
        } else {
            vizDiv.innerHTML = '<p>Plotly.js not loaded. Heatmap cannot be displayed.</p>';
        }
    }

}); // End DOMContentLoaded 