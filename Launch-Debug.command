#!/bin/bash

# Local AI Image Gen - Debug Launcher
# Double-click to start with verbose debug logging enabled.
# Watch output live: tail -f logs/server.log

cd "$(dirname "$0")"
exec bash Launch.command --debug
