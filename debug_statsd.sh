#!/bin/bash

echo "=== DogStatsD Connection Test ==="
echo ""

echo "1. Check port 8125 is listening:"
lsof -i :8125 || echo "❌ Port 8125 not listening"

echo ""
echo "2. Check Docker port mapping:"
docker port datadog-agent

echo ""
echo "3. Send test packet:"
echo -n "debug.manual.test:999|c" | nc -u -w1 localhost 8125
sleep 5

echo ""
echo "4. Check agent received it:"
docker exec datadog-agent agent status 2>/dev/null | grep -A 15 "DogStatsD"

echo ""
echo "5. Check agent logs for our metric:"
docker logs datadog-agent 2>&1 | grep -i "debug.manual.test" | tail -5

echo ""
echo "6. Check recent packets:"
docker logs datadog-agent 2>&1 | grep -i "packet" | tail -10
