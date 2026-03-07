#!/bin/bash

# Ultra Fast Image Gen - Mac Launcher
# Double-click this file to start the app (React UI on :7860)
#
# Usage:
#   ./Launch.command               — build React UI (if needed) + start FastAPI server
#   ./Launch.command --gradio      — start legacy Gradio UI on :7860
#   ./Launch.command --dev         — start FastAPI :7861 + Vite dev server :5173

cd "$(dirname "$0")"

echo "============================================"
echo "       Ultra Fast Image Gen for Mac"
echo "============================================"
echo ""

# ── Find uv ──────────────────────────────────────────────────────────────────

UV=""
for uvpath in /opt/homebrew/bin/uv ~/.local/bin/uv ~/.cargo/bin/uv /usr/local/bin/uv; do
    if [ -x "$uvpath" ]; then UV="$uvpath"; break; fi
done

if [ -z "$UV" ]; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    for uvpath in ~/.local/bin/uv /opt/homebrew/bin/uv ~/.cargo/bin/uv /usr/local/bin/uv; do
        if [ -x "$uvpath" ]; then UV="$uvpath"; break; fi
    done
fi

if [ -z "$UV" ]; then
    echo "Could not find uv. Please install it: https://docs.astral.sh/uv/"
    read -rp "Press Enter to exit..."
    exit 1
fi

echo "Using uv: $UV"
echo ""

# ── Modes ─────────────────────────────────────────────────────────────────────

if [[ "$1" == "--gradio" ]]; then
    # ── Legacy Gradio UI ───────────────────────────────────────────────────
    echo "Starting Gradio UI on http://127.0.0.1:7860"
    echo "(Press Ctrl+C to stop)"
    echo ""
    (sleep 6 && open http://127.0.0.1:7860) &
    UV_PROJECT_ENVIRONMENT=venv "$UV" run python app.py

elif [[ "$1" == "--dev" ]]; then
    # ── Dev mode: FastAPI backend + Vite frontend ──────────────────────────
    echo "Dev mode: FastAPI on :7861 + Vite HMR on :5173"
    echo "Open http://localhost:5173 in your browser"
    echo "(Press Ctrl+C to stop both servers)"
    echo ""

    # Start FastAPI backend in background
    UV_PROJECT_ENVIRONMENT=venv "$UV" run python server.py --port 7861 &
    BACKEND_PID=$!

    # Start Vite dev server
    cd frontend && npm run dev
    DEV_EXIT=$?

    # Cleanup
    kill "$BACKEND_PID" 2>/dev/null
    exit $DEV_EXIT

else
    # ── Production mode: build if needed, then serve ───────────────────────
    echo "Production mode: FastAPI + React UI on http://127.0.0.1:7860"
    echo ""

    # Build frontend if dist is missing or stale
    DIST="frontend/dist/index.html"
    NEED_BUILD=false

    if [ ! -f "$DIST" ]; then
        NEED_BUILD=true
        echo "Frontend not built yet — building now..."
    fi

    if [ "$NEED_BUILD" = true ]; then
        if ! command -v node &>/dev/null; then
            echo "ERROR: Node.js is required to build the frontend."
            echo "Install Node.js from https://nodejs.org/ and try again."
            echo "Or run with --gradio to use the legacy Gradio UI."
            read -rp "Press Enter to exit..."
            exit 1
        fi

        echo "Installing frontend dependencies..."
        (cd frontend && npm install)

        echo "Building frontend..."
        (cd frontend && npm run build)

        if [ ! -f "$DIST" ]; then
            echo "Build failed — cannot continue."
            read -rp "Press Enter to exit..."
            exit 1
        fi

        echo "Build complete."
        echo ""
    fi

    echo "Starting server... Opening browser in 5 seconds."
    echo "(Press Ctrl+C to stop)"
    echo ""

    (sleep 5 && open http://127.0.0.1:7860) &
    UV_PROJECT_ENVIRONMENT=venv "$UV" run python server.py --port 7860
fi
