```bash
docker compose down
```

```bash
docker compose up --build
```

```bash
docker compose build k6
```

```bash
docker compose run k6
```

```bash
curl http://localhost:8080/health
```

```bash
curl -X POST http://localhost:8084/reset_episode
```

```bash
curl http://localhost:8080/episode_status
```

```bash
curl -X POST http://localhost:8080/set_episode_length \
  -H "Content-Type: application/json" \
  -d '{"episode_length": 5}'
```