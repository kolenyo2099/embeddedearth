#!/bin/bash

# EmbeddedEarth Installation Script using uv
# Sets up environment and installs custom DOFA-CLIP fork

echo "🌍 Setting up EmbeddedEarth environment..."

# 1. Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ 'uv' is not installed."
    echo "Please install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ 'uv' found."

# 2. Create virtual environment
echo "📦 Creating virtual environment (venv)..."
uv venv venv
source venv/bin/activate

# 3. Clone DOFA-CLIP for custom open_clip fork
if [ ! -d "DOFA-CLIP" ]; then
    echo "📥 Cloning DOFA-CLIP repository..."
    git clone https://github.com/xiong-zhitong/DOFA-CLIP.git
else
    echo "✅ DOFA-CLIP repository already present."
fi

# 4. Install custom open_clip fork
echo "🔧 Installing custom open_clip fork (DOFA compatible)..."
cd DOFA-CLIP/open_clip
uv pip install -e .
cd ../..

# 5. Install other requirements
echo "📚 Installing application dependencies..."
uv pip install -r requirements.txt

# 6. Install frontend dependencies
if command -v npm &> /dev/null; then
    echo "📦 Installing frontend dependencies..."
    (cd frontend && npm install)
    echo "✅ Frontend dependencies installed."
else
    echo "⚠️ npm not found. Skipping frontend dependency install."
    echo "   Install Node.js/npm to build the Svelte frontend."
fi

echo "🎉 Installation complete!"
echo ""
echo "🚀 Starting backend and frontend..."
python run.py &
BACKEND_PID=$!

if command -v npm &> /dev/null; then
    (cd frontend && npm run dev -- --host 0.0.0.0 --port 5173) &
    FRONTEND_PID=$!
    echo "✅ Frontend dev server started (PID: $FRONTEND_PID)."
else
    echo "⚠️ npm not found. Frontend dev server not started."
fi

echo ""
echo "✅ Backend API running at http://localhost:8501"
echo "✅ Frontend dev server running at http://localhost:5173"
echo ""
echo "To stop services:"
echo "  kill $BACKEND_PID ${FRONTEND_PID:-}"
