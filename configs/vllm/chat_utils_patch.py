# Patch for vLLM chat_utils.py - Fix malformed JSON in tool call arguments
#
# Problem: When tool call arguments contain malformed JSON, vLLM crashes
# with JSONDecodeError instead of gracefully handling it.
#
# Apply with:
# docker exec devstral-vllm-nightly sed -i 's/item\["function"\]\["arguments"\] = json.loads(content)/try:\n                            item["function"]["arguments"] = json.loads(content)\n                        except json.JSONDecodeError:\n                            item["function"]["arguments"] = {"raw": content}/' /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py

# Better: use this Python one-liner to patch
"""
docker exec devstral-vllm-nightly python3 -c "
import re
path = '/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py'
with open(path, 'r') as f:
    content = f.read()

old = '''                    if not isinstance(content, (dict, list)):
                        item[\"function\"][\"arguments\"] = json.loads(content)'''

new = '''                    if not isinstance(content, (dict, list)):
                        try:
                            item[\"function\"][\"arguments\"] = json.loads(content)
                        except json.JSONDecodeError:
                            # Fallback for malformed JSON from model/client
                            item[\"function\"][\"arguments\"] = {\"_raw\": content}'''

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('Patched successfully')
"
"""
