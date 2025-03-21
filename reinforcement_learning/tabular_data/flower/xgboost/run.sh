#!/bin/bash

echo "Starting server"
python server.py --num_clients=$1 &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 $1`; do
    echo "Starting client $i"
    python client.py  --id=${i} --num_clients=$1 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait