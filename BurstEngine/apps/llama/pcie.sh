#!/bin/bash

# 获取所有PCI设备的列表
lspci -vv | grep "PCI bridge" | while read line
do
    # 提取设备ID
    device_id=$(echo $line | cut -d' ' -f1)

    # 获取该设备的详细信息
    device_info=$(lspci -vv -s $device_id)

    # 提取并显示PCIe版本信息
    echo "$device_info" | grep -Eo "LnkCap:.*Speed.*Width" | while read lnkcap
    do
        echo "Device $device_id: $lnkcap"
    done
done

