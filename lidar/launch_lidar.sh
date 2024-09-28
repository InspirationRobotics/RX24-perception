# Check if kill argument is passed
if [ "$1" == "--kill" ]; then
    echo "Killing LiDAR"
    screen -S combined_lidar -X quit
    screen -S livox_lidar -X quit
    exit 0
fi

# Check if screens are already running
if screen -ls | grep -q "livox_lidar"; then
    echo "LiDAR already running"
    exit 0
fi

# Launch combined lidar node
screen -dm -S combined_lidar bash -c "ros2 run combine_lidar combine_lidar_node"

# Check if --rviz argument is passed
if [ "$1" == "--rviz" ]; then
    echo "Launching LiDAR with RViz"
    screen -dm -S livox_lidar bash -c "ros2 launch livox_ros2_driver RX24_launch_rviz.py"
    exit 0
fi
echo "Launching LiDAR without RViz"
screen -dm -S livox_lidar bash -c "ros2 launch livox_ros2_driver RX24_launch.py"