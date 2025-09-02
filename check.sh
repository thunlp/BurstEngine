vailable_nodes_file="/tmp/available_nodes.tmp"
failed_nodes_file="/tmp/failed_nodes.tmp"
invalid_result_nodes_file="/tmp/invalid_result_nodes.tmp"
ssh_failed_nodes_file="/tmp/ssh_failed_nodes.tmp"
available_nodes_file="/tmp/avail_nodes.tmp"
oom_nodes_file="/tmp/oom_nodes.tmp"
node_prefix="bjdx"
prefix_len=${#node_prefix}
# 定义一个清理函数，在脚本退出时删除临时文件
cleanup() {
    rm -f "$available_nodes_file" "$failed_nodes_file" "$invalid_result_nodes_file" "$ssh_failed_nodes_file" "$oom_nodes_file"
}
trap cleanup EXIT

# 在脚本开始时删除任何已有的临时文件
cleanup

# 定义排除的节点列表
excluded_nodes=("")

# 节点连续化处理函数
process_continuous_nodes() {
    local nodes=($(cat "$1" | sort -V))
    local output=""
    local start_node=""
    local end_node=""

    for ((i = 0; i < ${#nodes[@]}; i++)); do
        local current_node=${nodes[$i]}
        local current_num=$(echo ${current_node:$prefix_len} | sed 's/^0*//') # 去除前导0

        # 初始化起始节点
        if [[ -z $start_node ]]; then
            start_node=$current_num
            end_node=$current_num
            continue
        fi

        # 检查当前节点是否连续
        echo $current_num
        if [[ $((current_num - end_node)) -eq 1 ]]; then
            end_node=$current_num
        else
            # 输出前一段连续的节点
            if [[ $start_node == $end_node ]]; then
                output+="${node_prefix}${start_node} "
            else
                output+="${node_prefix}${start_node}-${node_prefix}${end_node}"
            fi
            start_node=$current_num
            end_node=$current_num
        fi
    done

    # 处理最后一段节点
    if [[ $start_node == $end_node ]]; then
        output+="${node_prefix}$start_node"
    else
        output+="${node_prefix}$start_node-${node_prefix}$end_node"
    fi

    echo "$output"
}

# 循环从g01到g91
total=0
#for i in {12,} {41,} {47..49} {70..75}; do
#for i in  {30..39} {41,} {47..49} {52..54} {60..75} {82..90}; do
#g[43,45,69-75,79-81]
#for i in  {26..29} {30,} {31,} {32..33} {34,} {35,} {36..39} {41,} {43,} {47..49} {79..83} {84..86} {87..88} {89..91}; do
for i in  {1..2}; do
  echo $i

    node="${node_prefix}$i"
    # 检查当前节点是否在排除列表中
    if [[ " ${excluded_nodes[@]} " =~ " ${node} " ]]; then
        # 跳过该节点
        continue
    fi

    (
        # 通过ssh执行nvidia-smi命令，并检查所有显卡的显存是否为0
        result=$(ssh -o ConnectTimeout=1 -o StrictHostKeyChecking=no $node "nvidia-smi --query-gpu=memory.used --format=csv,noheader" 2>/dev/null)
        # 检查SSH连接和命令执行是否成功
        if [ $? -ne 0 ] || [ -z "$result" ]; then
            echo "$node" >> "$ssh_failed_nodes_file"
            continue
        fi

        # 处理命令输出并计算显存总占用
        used_memory=$(echo "$result" | awk '{sum += $1} END {print sum}')

        # 检查 used_memory 是否为空
        if [ -z "$used_memory" ]; then
            echo "$node" >> "$failed_nodes_file"
            continue
        fi

        # 检查结果是否为数值
        if ! [[ "$used_memory" =~ ^[0-9]+$ ]]; then
            echo "$node" >> "$invalid_result_nodes_file"
            continue
        fi

        # 如果总显存占用低于100MB，则认为节点可用，并记录到临时文件中
        if [ "$used_memory" -lt 100 ]; then
            echo "$node" >> "$available_nodes_file"

        else
            echo "$node" >> "$oom_nodes_file"
        fi
    ) &
done

# 等待所有后台进程完成
sleep 5
# 输出所有可用节点
if [ "$1" = "-l" ]; then
    if [ -s "$available_nodes_file" ]; then
        cat "$available_nodes_file"
    fi
    exit 0
fi

if [ -s "$available_nodes_file" ]; then
    total=`wc -l $available_nodes_file| awk '{print $1}'`
    echo "Available nodes: $(process_continuous_nodes "$available_nodes_file")"
    echo "Total avail: $total"
fi

# 输出SSH连接失败的节点
if [ -s "$ssh_failed_nodes_file" ]; then
    echo "Nodes with SSH connection failure: $(process_continuous_nodes "$ssh_failed_nodes_file")"
fi

# 输出检测到无效结果的节点
if [ -s "$invalid_result_nodes_file" ]; then
    echo "Nodes with invalid results: $(process_continuous_nodes "$invalid_result_nodes_file")"
fi

# 输出检测到其他错误的节点
if [ -s "$failed_nodes_file" ]; then
    echo "Nodes with other errors: $(process_continuous_nodes "$failed_nodes_file")"
fi

if [ -s "$oom_node_file" ]; then
    echo "Nodes are using by others: $(process_continuous_nodes "$failed_nodes_file")"
fi


