

# 1. 下载python环境
pip install -r multimodal/env/requirement.txt

# 2. 准备数据集
可以将data文件夹移动到autodl-tmp下，因为后续生成的数据集比较大

在/data/dataset文件下有三个数据集的文件夹。

执行每个文件夹内的clean.ipynb。

可能得到:
1. 归一化后的图像，用于生成完整的可视化结果:
   - HSI_norm.mat 归一化后的HSI图像
   - LIDAR_norm.mat 归一化后的LiDAR图像
   - SAR_norm.mat 归一化后的SAR图像
   - DSM_norm.mat 归一化后的DSM图像
 
2. 各种模态的测试集图像块、训练集图像块
   - HSI_Te.mat HSI测试集图像块
   - HSI_Te.mat HSI测试集图像块
   - HSI_Tr.mat HSI训练集图像块
   - LIDAR_Te.mat LiDAR测试集图像块
   - LIDAR_Tr.mat LiDAR训练集图像块
   - SAR_Te.mat SAR测试集图像块
   - SAR_Tr.mat SAR训练集图像块
   - DSM_Te.mat DSM测试集图像块
   - DSM_Tr.mat DSM训练集图像块

3. label
   - TeLabel.mat 测试集的label
   - TrLabel.mat 训练集集的label

对于三个数据集，各自的划分方式分别为：
## Trento数据集
MFT论文中采用的是disjoint划分，但这个数据集我只找到了整张图片的数据

此项目采用了sample_gt函数 每类选择184个测试样本的划分方式。

## MUUFL数据集
来自仓库https://github.com/GatorSense/MUUFLGulfport
训练集和测试集的划分在MFT论文中采用了5%的训练集和95%的测试集。
>  The table represents class-specific land-cover types and the number of randomly selected (5%) training and the remaining (95%) test samples

此项目采用sample_gt函数 同样随机选择5%的点来划分

## Augsburg数据集
来自仓库https://github.com/danfenghong/ISPRS_S2FL
训练集和测试集的划分在数据集中已经给出
训练集为TrainImage，测试集为TestImage
注意：这个数据集太大了，所以这个数据集的patch size 此项目设置成7 * 7

# 3. 配置邮箱（可选）
如果要使用邮箱接收运行结果：
   1. 申请邮箱授权码（[如何获得qq邮箱授权码](https://zhuanlan.zhihu.com/p/668505100)）

   2. 修改sendmail.py：
      - sendAddr,recipientAddrs=邮箱
      - password=授权码

   3. 取消注释test.sh文件中的倒数第一行和倒数第三行
      ```shell
      # python sendmail.py --title "finished : ${my_array[$i]} ($a/$sum)" --content "$result"
      # python sendmail.py --content "ALL FINISHED!" --file "YES" --title "AUTODL ALL $sum TASK FINISHED"
      ```

# 4. 运行
利用test.sh可以批量化运行一系列不同配置的实验。

## 运行步骤：
1. 在paramConfig.ini中，将[DEFAULT]的parent_directory改为data文件夹的新地址，例如/root/autodl-tmp/data
2. 在paramConfig.ini中，新建要跑的配置。我已经整理好了一些配置，可以参考起来设置新的配置，配置名不能重复。
3. 将要跑的配置名复制到test.sh中的my_array=()，使用回车分行。
3. `bash ./test.sh`  运行 ，` bash ./test.sh && /usr/bin/shutdown `  运行后关机

## 在代码中获取paramConfig.ini中配置的方法：

**boolean**：section.getboolean('HSIOnly')

**int**：int(section['token_num'])

**str**：section['checkpointName']

**tuple**: getResolution(section["kernel_size"])

## 其他注意事项：
整套代码都在multimodal.ipynb中，批量化运行时是使用test.sh将ipynb转换为py文件进行运行的，后续如果更改代码，也要在ipynb中更改。

如果想用jupyter运行代码进行调试，需要取消注释这一行，默认使用TEST中的配置运行。使用jupyter进行调试代码可以节省很多时间。
```jupyter
# # # jupyter需要下面一行
# sys.argv = ['multimodal.py ']
```