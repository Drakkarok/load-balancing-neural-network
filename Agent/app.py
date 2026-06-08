import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from requests.adapters import HTTPAdapter

app = Flask(__name__)


class LoadBalancerAgent:
    def __init__(self):
        self.servers = [
            {"id": "server-1", "url": "http://lbnn-server-1:8081"},
            {"id": "server-2", "url": "http://lbnn-server-2:8082"},
            {"id": "server-3", "url": "http://lbnn-server-3:8083"},
        ]
        self.server_states = {}
        self.current_tick = 0
        self.state_lock = threading.Lock()

        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)

        self.initialize_server_states()

    def initialize_server_states(self):
        for server in self.servers:
            try:
                response = self.session.get(f"{server['url']}/metrics", timeout=10)
                if response.status_code == 200:
                    self.server_states[server["id"]] = response.json()
                else:
                    self.server_states[server["id"]] = {
                        "cpu": 0.0, "memory": 0.0, "connections": 0, "tick": 0,
                        "bracket_counts": {
                            "cpu":   {"low": 0.0, "mid": 0.0, "high": 0.0},
                            "mem":   {"low": 0.0, "mid": 0.0, "high": 0.0},
                            "count": {"low": 0,   "mid": 0,   "high": 0},
                        },
                    }
            except Exception as e:
                print(f"Failed to initialize {server['id']}: {e}")
                self.server_states[server["id"]] = {
                    "cpu": 0.0, "memory": 0.0, "connections": 0, "tick": 0
                }

    def send_synchronized_requests(self, request_data, chosen_server_index):
        """Advance all servers one tick; only the chosen server receives the request."""
        self.current_tick += 1
        tick_id = self.current_tick

        def send_to_server(i, server):
            is_chosen = (i == chosen_server_index)
            payload = {
                "tick_id": tick_id,
                "request": request_data if is_chosen else None,
                "is_real": is_chosen,
            }
            try:
                t0 = time.time()
                response = self.session.post(
                    f"{server['url']}/process_request", json=payload, timeout=30
                )
                if time.time() - t0 > 1.0:
                    print(f"WARNING: {server['id']} took {time.time()-t0:.2f}s")
                if response.status_code == 200:
                    return server["id"], response.json()
                print(f"Error from {server['id']}: {response.status_code}")
                return server["id"], None
            except Exception as e:
                print(f"Failed to contact {server['id']}: {e}")
                return server["id"], None

        responses = {}
        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {
                executor.submit(send_to_server, i, server): server
                for i, server in enumerate(self.servers)
            }
            for future in as_completed(futures):
                sid, result = future.result()
                if result:
                    responses[sid] = result

        return responses, tick_id

    def update_server_states(self, responses):
        with self.state_lock:
            for server_id, response in responses.items():
                if "current_state" in response:
                    self.server_states[server_id] = response["current_state"]

    def get_current_server_states(self):
        with self.state_lock:
            return self.server_states.copy()

    def reset_episode(self):
        self.current_tick = 0
        for server in self.servers:
            try:
                self.session.get(f"{server['url']}/reset_episode", timeout=10)
            except Exception as e:
                print(f"Failed to reset {server['id']}: {e}")
        self.initialize_server_states()


agent = LoadBalancerAgent()


@app.route("/step_training", methods=["POST"])
def step_training():
    data = request.get_json()
    request_data = data.get("request")
    forced_action = data.get("forced_action")

    if request_data is None or forced_action is None:
        return {"error": "Missing request or forced_action"}, 400

    chosen_server_index = int(forced_action)
    chosen_server = agent.servers[chosen_server_index]

    responses, tick_id = agent.send_synchronized_requests(request_data, chosen_server_index)
    agent.update_server_states(responses)

    return {
        "status": "processed",
        "tick_id": tick_id,
        "chosen_server": chosen_server["id"],
        "current_server_states": agent.get_current_server_states(),
    }


@app.route("/reset_episode", methods=["GET", "POST"])
def reset_episode():
    agent.reset_episode()
    return {"status": "episode_reset", "tick": agent.current_tick}


@app.route("/server_states")
def get_server_states():
    return jsonify(agent.get_current_server_states())


if __name__ == "__main__":
    print("Load Balancer Agent Starting...")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
