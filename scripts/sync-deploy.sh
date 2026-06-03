#!/usr/bin/env bash
# Stage runtime files into deploy/ for AgentCore Container builds.
# Run before: agentcore deploy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$ROOT/deploy"

mkdir -p "$DEPLOY/riskclassifier_v2"

# Clean prior staging but keep Dockerfile
for item in "$DEPLOY"/*; do
  base="$(basename "$item")"
  case "$base" in
    Dockerfile) ;;
    *) rm -rf "$item" ;;
  esac
done

cp "$ROOT/requirements-agentcore.txt" "$DEPLOY/requirements-agentcore.txt"
cp "$ROOT/deploy/Dockerfile" "$DEPLOY/Dockerfile" 2>/dev/null || cp "$ROOT/Dockerfile" "$DEPLOY/Dockerfile"

rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$ROOT/agentcore_main.py" \
  "$ROOT/agents" \
  "$ROOT/tools" \
  "$ROOT/backend" \
  "$ROOT/config" \
  "$ROOT/conversation_history" \
  "$DEPLOY/"

rsync -a "$ROOT/riskclassifier_v2/crisis_response.py" "$DEPLOY/riskclassifier_v2/"

echo "Staged slim AgentCore runtime into $DEPLOY (run agentcore deploy next)"
