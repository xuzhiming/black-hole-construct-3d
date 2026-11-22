#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)/web"
echo "Starting Black Hole Construct 3D Preview at http://127.0.0.1:8765/"
python3 -m http.server 8765 --directory "$DIR" --bind 127.0.0.1
