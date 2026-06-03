#!/usr/bin/env bash
# Full AgentCore deploy: stage slim image context, optional stack cleanup, deploy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

"$ROOT/scripts/sync-deploy.sh"

if [[ "${DELETE_FAILED_STACK:-}" == "1" ]]; then
  "$ROOT/scripts/delete-agentcore-stack.sh"
fi

rm -rf agentcore/.cache agentcore/cdk/cdk.out .cdk-out

agentcore deploy --target "${AGENTCORE_TARGET:-default}" --yes
