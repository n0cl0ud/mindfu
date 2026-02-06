#!/bin/bash
# Entrypoint script that patches vLLM before starting the server
# Fixes:
# 1. JSONDecodeError when tool call arguments contain malformed JSON
# 2. IndexError when streamed_args_for_tool index is out of range

CHAT_UTILS="/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py"
SERVING="/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"

# =============================================================================
# Patch 1: chat_utils.py - JSONDecodeError fix
# =============================================================================
if ! grep -q "_raw" "$CHAT_UTILS" 2>/dev/null; then
    echo "[entrypoint-patch] Applying chat_utils.py JSON error handling patch..."
    python3 -c "
path = '$CHAT_UTILS'
with open(path, 'r') as f:
    content = f.read()

old = '''                    if not isinstance(content, (dict, list)):
                        item[\"function\"][\"arguments\"] = json.loads(content)'''

new = '''                    if not isinstance(content, (dict, list)):
                        try:
                            item[\"function\"][\"arguments\"] = json.loads(content)
                        except json.JSONDecodeError:
                            item[\"function\"][\"arguments\"] = {\"_raw\": content}'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('[entrypoint-patch] chat_utils.py patch applied successfully')
else:
    print('[entrypoint-patch] chat_utils.py pattern not found, skipping')
"
else
    echo "[entrypoint-patch] chat_utils.py patch already applied, skipping"
fi

# =============================================================================
# Patch 2: serving.py - IndexError fix for streamed_args_for_tool
# =============================================================================
if ! grep -q "len(tool_parser.streamed_args_for_tool)" "$SERVING" 2>/dev/null; then
    echo "[entrypoint-patch] Applying serving.py IndexError patch..."
    python3 -c "
path = '$SERVING'
with open(path, 'r') as f:
    content = f.read()

old = '''                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]'''

new = '''                            # get what we've streamed so far for arguments
                            # for the current tool
                            if index < len(tool_parser.streamed_args_for_tool):
                                actual_call = tool_parser.streamed_args_for_tool[index]
                            else:
                                actual_call = \"\"'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('[entrypoint-patch] serving.py patch applied successfully')
else:
    print('[entrypoint-patch] serving.py pattern not found, skipping')
"
else
    echo "[entrypoint-patch] serving.py patch already applied, skipping"
fi

# Start vLLM with all arguments passed to this script
exec python3 -m vllm.entrypoints.openai.api_server "$@"
