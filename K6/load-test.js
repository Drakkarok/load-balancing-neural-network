import http from "k6/http";
import { check, sleep } from "k6";

function randInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function normalInt(mean, std, min, max) {
    // Box-Muller transform
    const u = 1 - Math.random();
    const v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return Math.min(max, Math.max(min, Math.round(mean + z * std)));
}

function generateRequest() {
    const types = ["light", "medium", "heavy"];
    const type = types[Math.floor(Math.random() * types.length)];
    if (type === "light") {
        return { type, cpu_cost: randInt(150, 200), memory_cost: randInt(100, 125), duration: normalInt(5.0, 2.0, 1, 9) };
    } else if (type === "medium") {
        return { type, cpu_cost: randInt(350, 400), memory_cost: randInt(300, 350), duration: normalInt(10.5, 2.5, 5, 16) };
    } else {
        return { type, cpu_cost: randInt(400, 500), memory_cost: randInt(400, 450), duration: normalInt(16.5, 2.0, 12, 21) };
    }
}

export const options = {
    stages: [
        { duration: "40s", target: 1 }, // Single user for testing
    ],
};

export default function () {
    const requestType = generateRequest();

    console.log(
        `Sending ${requestType.type} request: CPU=${requestType.cpu_cost}, Memory=${requestType.memory_cost}, Duration=${requestType.duration}`
    );

    // Send request to agent
    const payload = {
        request: requestType,
    };

    const response = http.post(
        "http://lbnn-agent:8080/route_request",
        JSON.stringify(payload),
        {
            headers: { "Content-Type": "application/json" },
            timeout: "30s",
        }
    );

    // Check if request was successful
    check(response, {
        "status is 200": (r) => r.status === 200,
        "response has tick_id": (r) => JSON.parse(r.body).tick_id !== undefined,
        "response has chosen_server": (r) =>
            JSON.parse(r.body).chosen_server !== undefined,
    });

    if (response.status === 200) {
        const data = JSON.parse(response.body);
        console.log(`Tick ${data.tick_id}: Routed to ${data.chosen_server}`);
        console.log(
            `Server states: ${JSON.stringify(data.current_server_states)}`
        );
    } else {
        console.error(`Request failed: ${response.status} - ${response.body}`);
    }

    // Wait 2 seconds between requests to see state changes clearly
    sleep(2);
}
