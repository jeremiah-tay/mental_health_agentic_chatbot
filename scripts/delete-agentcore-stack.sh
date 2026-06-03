#!/usr/bin/env bash
# Delete a failed AgentCore CloudFormation stack (e.g. ROLLBACK_COMPLETE) before redeploy.
set -euo pipefail

STACK_NAME="${AGENTCORE_STACK_NAME:-AgentCore-mentalhealthchatbot-default}"
REGION="${AWS_REGION:-ap-southeast-1}"

echo "Deleting stack $STACK_NAME in $REGION ..."
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"
echo "Stack deleted."
