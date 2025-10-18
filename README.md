# Inference Engine - Standalone AI-Powered Model Analysis

A modern, standalone inference engine with a beautiful Textual TUI for comparing original vs modified model outputs using AI-powered tensor modifications.

## Features

- 🎨 **Beautiful Textual TUI** - Modern terminal UI with tabs, panels, and real-time updates
- 🤖 **AI-Powered Suggestions** - Uses AI agents to suggest optimal tensor modifications
- 🔧 **Hook-Based Modifications** - Apply tensor modifications during inference without changing model files
- ⚡ **Real-time Comparison** - Side-by-side view of original vs modified outputs
- 📊 **Multiple Configurations** - Preset configs for different use cases (conservative, creative, mathematical, etc.)
- 🧠 **Qwen Model Support** - Special handling for thinking tokens in Qwen models

## Architecture

```
inference_standalone/
├── inference_engine.py      # Main Textual app
├── model_loader.py          # Model loading and inference
├── hook_system.py           # Hook-based tensor modifications
├── config_manager.py        # Inference configurations
├── ai_agent.py              # Centralized AI agent (smolagents)
├── server/
│   └── api.py               # FastAPI service exposing inference endpoints
├── tensor_inspector.py      # Tensor analysis with AI (coming soon)
├── ai_suggestion_engine.py  # AI suggestions with ROME (coming soon)
├── rome_analyzer.py         # ROME analysis (coming soon)
└── ui/
    ├── comparison_widget.py # Side-by-side comparison widget
    └── ...                  # More UI components

frontend/
├── src/App.tsx              # React frontend replicating TUI features
├── src/api.ts               # API helpers for backend communication
└── ...
```

## Installation

### 1. Clone or copy the inference_standalone directory

```bash
cd inference_standalone
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the directory:

```env
# OpenRouter API for AI features
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet  # or your preferred model
```

## Usage

### Web Frontend (Recommended)

The new React frontend talks to the FastAPI service for a richer experience.

1. Install backend dependencies (see [Installation](#installation)).
2. Start the API server (defaults to the bundled `Qwen_0.6B` weights):

   ```bash
   uvicorn server.api:app --reload
   ```

   Set `MODEL_PATH=/path/to/model` in the environment before running the server to point at a different model.

3. Install and run the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Vite proxies `/api/*` requests to the local FastAPI instance. Build with `npm run build` for production.

### Legacy Textual TUI

The original terminal UI remains available if you prefer a purely terminal workflow.

```bash
python inference_engine.py /path/to/your/model
```

Example:

```bash
python inference_engine.py ../models/Qwen2.5-7B-Instruct
```

## UI Overview

### Main Screen Layout

```
┌─────────────────────────────────────────────────────────┐
│ Inference Engine                              12:34:56  │
├─────────────────────────────────────────────────────────┤
│ [Comparison] [Configurations] [AI Suggestions] [History]│
├─────────────────────────────────────────────────────────┤
│ Query: [Enter your query here...]                       │
├─────────────────────────────────────────────────────────┤
│ Config: [Balanced ▼] [Generate] [Apply] [Clear] [AI]   │
├──────────────────────┬──────────────────────────────────┤
│ 🔵 Original Response │ 🔧 Modified (10 hooks active)    │
│                      │                                  │
│ [Original text...]   │ [Modified text with hooks...]    │
│                      │                                  │
├──────────────────────┴──────────────────────────────────┤
│ Status: Ready. Model loaded on cuda.                    │
└─────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

- `Ctrl+G` - Generate responses
- `Ctrl+C` - Clear comparison view
- `Ctrl+S` - Get AI suggestions
- `Ctrl+H` - Toggle hooks on/off
- `Ctrl+Q` - Quit application
- `Tab` - Navigate between UI elements

## Workflow

### 1. Basic Comparison

1. Enter a query in the input field
2. Select a configuration preset (or use default "Balanced")
3. Click "Generate" or press `Ctrl+G`
4. View the original response in the left panel

### 2. Apply AI Modifications

1. Click "Get AI Suggestions" to generate tensor modifications
2. Review suggestions in the "AI Suggestions" tab
3. Click "Apply Suggestions" to activate hooks
4. Generate again to see modified output in right panel

### 3. Configuration Presets

Available presets:
- **Conservative** - Low temperature for logical tasks
- **Balanced** - Default for general use
- **Creative** - Higher temperature for creative writing
- **Mathematical** - Optimized for math problems
- **Coding** - Precise settings for code generation
- **Fast** - Quick responses with fewer tokens

## API Configuration

The AI features require an OpenRouter API key. Get one from [OpenRouter](https://openrouter.ai/).

Supported models:
- `anthropic/claude-3.5-sonnet` (recommended)
- `anthropic/claude-3-opus`
- `openai/gpt-4-turbo`
- Any other OpenRouter-supported model

## Extending the System

### Adding New Configurations

Edit `config_manager.py` to add new presets:

```python
"custom_name": InferenceConfig(
    name="Custom Config",
    description="Your description",
    temperature=0.7,
    top_p=0.95,
    # ... other parameters
)
```

### Custom Hook Operations

Edit `hook_system.py` to add new tensor operations:

```python
elif operation == "your_operation":
    modified_output = self._apply_your_operation(modified_output, value)
```

## Troubleshooting

### Model won't load
- Check the model path exists
- Ensure you have enough RAM/VRAM
- Verify transformers library is installed

### AI suggestions not working
- Check your OPENROUTER_API_KEY in .env
- Verify internet connection
- Check API quota/credits

### UI rendering issues
- Update textual: `pip install --upgrade textual`
- Try a different terminal emulator
- Check terminal supports Unicode

## Future Enhancements

- [ ] Complete tensor_inspector.py with full analysis
- [ ] Add ROME-enhanced suggestion engine
- [ ] Stream responses token by token
- [ ] Export comparison results
- [ ] Save/load modification presets
- [ ] Batch processing mode

## License

MIT License - See LICENSE file for details

## Credits

Built on top of the Tensor Slayer project, leveraging:
- Textual for the beautiful TUI
- Transformers for model operations
- Smolagents for AI capabilities
- Rich for terminal formatting
