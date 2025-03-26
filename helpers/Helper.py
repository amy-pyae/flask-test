import hashlib

#Need to modify
def generate_agent_key(data: str):
    return hashlib.sha256(data.encode()).hexdigest()[:16]  # Fixed 16 chars