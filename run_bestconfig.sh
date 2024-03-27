#!/bin/bash

projects=(        
          "bcp1_6000-bcp2_6000-exp0050" \
          "bcp2_6000-bcp1_6000-exp0050" \
          "bcp1_6000-bcp3_6000-exp0050" \
          "bcp3_6000-bcp1_6000-exp0050" \
          "bcp2_6000-bcp3_6000-exp0050" \
          "bcp3_6000-bcp2_6000-exp0050" \
          "bcp1_6000-pcp1_6000-exp0050" \
          "pcp1_6000-bcp1_6000-exp0050" \
          "bcp1_6000-mp1_6000-exp0050" \
          "mp1_6000-bcp1_6000-exp0050"          
          )

for i in ${projects[@]}; do        
    proj=$i  # 要查找的proj名称
    max_acc=0  # 最高准确率初始化为0
    max_acc_config=""  # 保存最高准确率对应的配置文件名

    # 遍历config目录下的所有json文件
    for file in config/*.json; do
        # 获取文件名（不包含目录部分）
        filename=$(basename "$file")
        
        # 提取proj名称和准确率
        proj_name=$(echo "$filename" | grep -oE ".*_acc" | sed 's/_acc//' | sed 's/-*$//')
        accuracy=$(echo "$filename" | grep -oE "acc_[0-9]+\.[0-9]+")

        # 检查是否是目标proj的文件
        if [[ "$proj_name" == "$proj" ]]; then
            # 提取准确率数值
            acc_value=$(echo "$accuracy" | grep -oE "[0-9]+\.[0-9]+")
            
            # 比较准确率，更新最高准确率和对应的配置文件名
            if (( $(echo "$acc_value > $max_acc" | bc -l) )); then
                max_acc="$acc_value"
                max_acc_config="$filename"
            fi
        fi
    done

    # 输出最高准确率和对应的配置文件名
    echo "最高准确率：$max_acc"
    echo "对应的配置文件：$max_acc_config"

    # 删除其他准确率较低的配置文件
    for file in config/*.json; do
        filename=$(basename "$file")
        proj_name=$(echo "$filename" | grep -oE ".*_acc" | sed 's/_acc//' | sed 's/-*$//')
        if [[ "$proj_name" == "$proj" ]]; then
            if [[ "$filename" != "$max_acc_config" ]]; then
                rm "$file"
            fi
        fi        
    done
    
done

