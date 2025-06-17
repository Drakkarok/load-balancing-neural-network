import http from "k6/http";
import { check, sleep } from "k6";

// Define request types with different load characteristics
const requestTypes = [
    // ! Redefine these so they make sense ~ 10 cpu => 10000 resursa cpu total, fa-le sa varieze random intr-un interval. Ca sa ai cele 3 cazuri
    // maybe pot varia si duratia si light medium hard sa fie overall, si sa am alte doua taguri, resource intensive si duration,
    // resurce intensive (high) + duration (high) => hard task. un fel de sistem in care sa am mai multe tagguri. poate ma ajuta la partea
    // de ML sa am mai multe taguri sau sa fie mai granulare.
    { type: "light", cpu_cost: 50, memory_cost: 30, duration: 2 },
    { type: "medium", cpu_cost: 150, memory_cost: 100, duration: 4 },
    { type: "heavy", cpu_cost: 300, memory_cost: 200, duration: 6 },
];

export const options = {
    stages: [
        { duration: "40s", target: 1 }, // Single user for testing
    ],
};

export default function () {
    // Pick a random request type
    const requestType =
        requestTypes[Math.floor(Math.random() * requestTypes.length)];

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
