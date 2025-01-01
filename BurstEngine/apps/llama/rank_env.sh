#!/bin/bash

# Define the list of nodes
node_list=("g3011" "g3015" "g3019" "g3023")

# Get the hostname of the current machine
hostname=$(hostname)
endpoint="${node_list[0]}:7778"

# Find the index of the hostname in the node list
rank=-1
for i in "${!node_list[@]}"; do
   if [[ "${node_list[$i]}" = "$hostname" ]]; then
       rank=$i
       break
   fi
done

# Check if the hostname was found in the node list
if [[ $rank -eq -1 ]]; then
    echo "Hostname not found in the node list"
else
    echo "Rank of the hostname ('$hostname') in the node list is: $rank"
    echo "Endpoint of the hostname ('$hostname') in the node list is: $endpoint"
fi

export rank=$rank
export endpoint=$endpoint
