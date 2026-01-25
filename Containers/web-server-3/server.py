import os
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)
server_id = os.environ.get('SERVER_ID', 'unknown')

# Server capacity configurations (different for each server)
server_configs = {
    "server-3": {"max_cpu": 5000, "max_memory": 3000}   # Larger CPU, less memory
}

# Simulated server state
server_state = {
    "cpu_usage": 0,
    "memory_usage": 0, 
    "active_connections": 0,
    "active_requests": [],  # Requests currently consuming resources
    "request_history": [],  # Complete tick-by-tick history
    "current_tick": 0
}

def update_to_tick(tick_id):
    """Update server state to the current tick, removing expired requests"""
    # Remove requests that have expired by this tick
    server_state["active_requests"] = [
        req for req in server_state["active_requests"] 
        if req["expires_at_tick"] > tick_id
    ]
    
    # Update current tick
    server_state["current_tick"] = tick_id
    
    # Recalculate resource usage based on remaining active requests
    update_resource_usage()

def add_real_request(request_data, tick_id):
    """Add a new real request that will consume resources for N ticks"""
    duration_ticks = request_data.get('duration', 5)  # Duration in ticks
    cpu_cost = request_data.get('cpu_cost', 10) # Cost in adminesional units
    memory_cost = request_data.get('memory_cost', 5) # Cost in adminesional units
    
    # Add request that will expire after duration_ticks
    server_state["active_requests"].append({
        "tick_id": tick_id,
        "cpu_cost": cpu_cost,
        "memory_cost": memory_cost,
        "expires_at_tick": tick_id + duration_ticks,
        "request_type": request_data.get('type', 'unknown')
    })
    
    # Recalculate resource usage
    update_resource_usage()

def update_resource_usage():
    """Calculate current resource usage based on active requests and server capacity"""
    config = get_server_config()  
    
    total_cpu_load = sum(req["cpu_cost"] for req in server_state["active_requests"])
    total_memory_load = sum(req["memory_cost"] for req in server_state["active_requests"])
    
    # Calculate percentage usage based on server capacity
    server_state["cpu_usage"] = min(100, (total_cpu_load / config["max_cpu"]) * 100)
    server_state["memory_usage"] = min(100, (total_memory_load / config["max_memory"]) * 100)
    server_state["active_connections"] = len(server_state["active_requests"])

def get_current_metrics():
    """Get current server metrics"""
    return {
        "cpu": round(server_state["cpu_usage"], 2),
        "memory": round(server_state["memory_usage"], 2), 
        "connections": server_state["active_connections"],
        "tick": server_state["current_tick"]
    }
    
def get_server_config():
    """Get server config, fail fast if not found"""
    if server_id not in server_configs:
        raise ValueError(f"Error in fetching server config: server_id '{server_id}' not found in server_configs")
    return server_configs[server_id]

state_lock = threading.Lock()

@app.route('/health')
def health():
    try:
        config = get_server_config()
        return {
            "status": f"{server_id} is running",
            "config": config
        }
    except ValueError as e:
        return {"error": str(e)}, 500

@app.route('/process_request', methods=['POST'])
def process_request():
    data = request.get_json()
    tick_id = data.get('tick_id')
    request_data = data.get('request')
    is_real = data.get('is_real', True)
    
    with state_lock:
        # Update to current tick (removes expired requests)
        update_to_tick(tick_id)
        
        state_before_processing = get_current_metrics()

        if is_real and request_data:
            # Add new real request to active requests
            add_real_request(request_data, tick_id)
        
        # Record final state after processing this tick
        state_after_processing = get_current_metrics()
        
        # Log this tick in history
        server_state["request_history"].append({
            "tick_id": tick_id,
            "request": request_data if is_real else None,
            "state_after_processing": state_before_processing,
            "processed_real_request": is_real and request_data is not None
        })
    
    return {"status": "processed", "server_id": server_id, "tick_id": tick_id, "current_state": state_after_processing}

@app.route('/metrics')
def get_metrics():
    """Expose current metrics"""
    with state_lock:
        return jsonify(get_current_metrics())

@app.route('/get_episode_history')
def get_episode_history():
    """Return complete episode history for training"""
    with state_lock:
        return jsonify({
            "server_id": server_id,
            "episode_data": server_state["request_history"],
            "server_config": get_server_config()
        })

@app.route('/reset_episode')
def reset_episode():
    """Reset server state for new episode"""
    with state_lock:
        server_state["active_requests"] = []
        server_state["request_history"] = []
        server_state["current_tick"] = 0
        update_resource_usage()
        return {"status": "reset", "server_id": server_id}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False, threaded=True)