import requests
import threading
from flask import Flask, request, jsonify
from requests.adapters import HTTPAdapter

app = Flask(__name__)

class LoadBalancerAgent:
    def __init__(self):
        self.servers = [
            {"id": "server-1", "url": "http://lbnn-server-1:8081"},
            {"id": "server-2", "url": "http://lbnn-server-2:8082"},  
            {"id": "server-3", "url": "http://lbnn-server-3:8083"}
        ]
        self.server_states = {}  # Current state of each server
        self.current_tick = 0
        self.state_lock = threading.Lock()
        
        # Simple round robin for now (will replace with ML later)
        self.round_robin_index = 0
        
        # Episode tracking
        self.episode_length = 20  # requests per episode (configurable)
        self.current_episode_requests = 0
        self.trainer_url = "http://lbnn-trainer:8084"
        
        # Initialize by getting current server states
        
        # Optimize connections to servers
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        
        self.initialize_server_states()
    
    def initialize_server_states(self):
        """Get initial server states at startup"""
        print("Initializing server states...")
        for server in self.servers:
            try:
                response = self.session.get(f"{server['url']}/metrics", timeout=10)
                if response.status_code == 200:
                    self.server_states[server['id']] = response.json()
                    print(f"Initialized {server['id']}: {response.json()}")
                else:
                    # Initialize with empty state structure
                    self.server_states[server['id']] = {
                        "cpu": 0.0, "memory": 0.0, "connections": 0, "tick": 0
                    }
            except Exception as e:
                print(f"Failed to initialize {server['id']}: {e}")
                # Fallback initialization
                self.server_states[server['id']] = {
                    "cpu": 0.0, "memory": 0.0, "connections": 0, "tick": 0
                }
    
    def make_routing_decision(self, request_data): # ! Change later from round robin to ML that takes into consideration the states of the servers
        """Decide which server to route request to (Round Robin for now)"""
        current_states = self.get_current_server_states()
        
        # Simple round robin - replace with ML later
        chosen_server_index = self.round_robin_index % len(self.servers)
        self.round_robin_index += 1
        
        chosen_server = self.servers[chosen_server_index]
        print(f"Routing decision: {chosen_server['id']} (Round Robin)")
        return chosen_server_index, chosen_server
    
    def send_synchronized_requests(self, request_data, chosen_server_index):
        """Send requests to all servers - real to chosen, empty to others"""
        self.current_tick += 1
        tick_id = self.current_tick
        
        responses = {}
        
        import time
        start_time = time.time()
        
        for i, server in enumerate(self.servers):
            # ... (loop content same)
            is_chosen = (i == chosen_server_index)
            
            payload = {
                "tick_id": tick_id,
                "request": request_data if is_chosen else None,
                "is_real": is_chosen
            }
            
            try:
                s_start = time.time()
                response = self.session.post(
                    f"{server['url']}/process_request",
                    json=payload,
                    timeout=30
                )
                duration = time.time() - s_start
                if duration > 1.0:
                    print(f"WARNING: {server['id']} took {duration:.2f}s to respond!")
                
                if response.status_code == 200:
                    responses[server['id']] = response.json()
                    print(f"Tick {tick_id}: {server['id']} responded: {response.json()}")
                else:
                    print(f"Error from {server['id']}: {response.status_code}")
                    
            except Exception as e:
                print(f"Failed to contact {server['id']}: {e}")
                
        total_duration = time.time() - start_time
        if total_duration > 2.0:
            print(f"SLOW TICK: Total sync took {total_duration:.2f}s")
        
        return responses, tick_id
    
    def update_server_states(self, responses):
        """Update internal server states from responses"""
        with self.state_lock:
            for server_id, response in responses.items():
                if 'current_state' in response:
                    self.server_states[server_id] = response['current_state']
                    print(f"Updated {server_id} state: {response['current_state']}")
    
    def get_current_server_states(self):
        """Get current states for decision making"""
        with self.state_lock:
            return self.server_states.copy()
    
    def trigger_trainer(self):
        """Trigger trainer when episode completes"""
        print(f"=== EPISODE {self.current_episode_requests} COMPLETE ===")
        print("Triggering trainer...")
        
        try:
            response = requests.post(
                f"{self.trainer_url}/start_training",
                json={
                    "episode_requests": self.current_episode_requests,
                    "episode_ticks": self.current_tick,
                    "triggered_by": "agent"
                },
                timeout=60  # Give trainer time to process
            )
            
            if response.status_code == 200:
                result = response.json()
                print("Trainer completed successfully!")
                print(f"Analysis: {result.get('analysis', {})}")
                return True
            else:
                print(f"Trainer failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error triggering trainer: {e}")
            return False
    
    def reset_episode(self, hard_reset=False):
        """Reset episode counter and optionally servers"""
        print("Resetting episode...")
        self.current_episode_requests = 0
        
        if hard_reset:
            print("Performing HARD RESET on servers...")
            self.current_tick = 0 
            for server in self.servers:
                try:
                    self.session.get(f"{server['url']}/reset_episode", timeout=10)
                except Exception as e:
                    print(f"Failed to reset {server['id']}: {e}")
            self.initialize_server_states()

        print(f"Ready for new episode (length: {self.episode_length})")

# Global agent instance
agent = LoadBalancerAgent()

@app.route('/health')
def health():
    return {
        "status": "Agent is running", 
        "current_tick": agent.current_tick,
        "episode_progress": f"{agent.current_episode_requests}/{agent.episode_length}"
    }

@app.route('/route_request', methods=['POST'])
def route_request():
    """Main endpoint for receiving requests to route"""
    data = request.get_json()
    request_data = data.get('request')
    
    if not request_data:
        return {"error": "No request data provided"}, 400
    
    # Make routing decision
    chosen_index, chosen_server = agent.make_routing_decision(request_data)
    
    # Send synchronized requests to all servers
    responses, tick_id = agent.send_synchronized_requests(request_data, chosen_index)
    
    # Update server states from responses
    agent.update_server_states(responses)
    
    # Track episode progress
    agent.current_episode_requests += 1
    
    # Check if episode complete
    episode_complete = False
    if agent.current_episode_requests >= agent.episode_length:
        print(f"\n=== EPISODE COMPLETE ({agent.episode_length} requests) ===")
        
        # Trigger trainer (this might take a while)
        trainer_success = agent.trigger_trainer()
        
        # Reset for next episode
        agent.reset_episode()
        episode_complete = True
    
    # Signal to k6 when Agent is ready for next request
    return {
        "status": "routed",
        "tick_id": tick_id,
        "chosen_server": chosen_server['id'], # Debugging purposes
        "server_responses": len(responses), # Debugging purposes
        "current_server_states": agent.get_current_server_states(), # Debugging purposes
        "episode_progress": f"{agent.current_episode_requests}/{agent.episode_length}",
        "episode_complete": episode_complete
    }

@app.route('/server_states')
def get_server_states():
    """Debug endpoint to see current server states"""
    return jsonify(agent.get_current_server_states())

@app.route('/episode_status')
def episode_status():
    """Check current episode progress"""
    return {
        "current_requests": agent.current_episode_requests,
        "episode_length": agent.episode_length,
        "progress_percentage": (agent.current_episode_requests / agent.episode_length) * 100,
        "current_tick": agent.current_tick
    }

@app.route('/set_episode_length', methods=['POST'])
def set_episode_length():
    """Configure episode length"""
    data = request.get_json()
    new_length = data.get('episode_length')
    
    if not isinstance(new_length, int) or new_length <= 0:
        return {"error": "episode_length must be a positive integer"}, 400
    
    agent.episode_length = new_length
    return {
        "status": "updated",
        "new_episode_length": agent.episode_length
    }

@app.route('/step_training', methods=['POST'])
def step_training():
    """
    Endpoint for RL Training step.
    Receives: { "request": {...}, "forced_action": int }
    Returns: Updated state after action.
    """
    data = request.get_json()
    request_data = data.get('request')
    forced_action = data.get('forced_action')
    
    if request_data is None or forced_action is None:
        return {"error": "Missing request or forced_action"}, 400
        
    # Execute specific action
    chosen_server_index = int(forced_action)
    chosen_server = agent.servers[chosen_server_index]
    
    # Send synchronized requests (Agent Logia)
    responses, tick_id = agent.send_synchronized_requests(request_data, chosen_server_index)
    
    # Update internal state
    agent.update_server_states(responses)
    
    # Track progress (optional during training, but good for consistency)
    agent.current_episode_requests += 1
    
    return {
        "status": "processed",
        "tick_id": tick_id,
        "chosen_server": chosen_server['id'],
        "current_server_states": agent.get_current_server_states()
    }

@app.route('/reset_episode', methods=['GET', 'POST']) # Allow POST for explicit calls
def manual_reset_episode():
    """Manual trigger for episode reset (used by RL Env)"""
    # Hard reset to ensure fresh state for new episode
    agent.reset_episode(hard_reset=True)
    return {"status": "episode_reset", "tick": agent.current_tick}

if __name__ == '__main__':
    print("Load Balancer Agent Starting...")
    print(f"Episode length: {agent.episode_length} requests")
    print("Will trigger trainer automatically when episodes complete")
    app.run(host='0.0.0.0', port=8080, debug=True)