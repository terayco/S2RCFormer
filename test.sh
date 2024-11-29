#  bash ./test.sh  运行
# bash ./test.sh && /usr/bin/shutdown   运行后关机
#  tensorboard --logdir=/root/autodl-tmp/data/runs/  --samples_per_plugin images=100 打开tensorboard
# 1:  q2k1v1,q1k2v2 
# 2:  q2k1v1
# 3:  q1k2v2
# 4:  q1k1v1,q2k1v1
# 5:  q1k1v1,q2k2v2
# 6:  q1k1v1,q2k2v2,q2k1v1,q1k2v2

#!/bin/bash
my_array=(
# TRAIN,Trento,pos
TRAIN,Trento
TRAIN,MUUFL
TRAIN,AugsburgSAR
TRAIN,AugsburgDSM
) 
echo $my_array
jupyter nbconvert --to script multimodal.ipynb
sum=${#my_array[@]}
echo "Total $sum configs... "
for((i=0;i<$sum;i++))
    do
        let a=i+1
        echo "excute config : ${my_array[$i]} ($a/$sum)" 
        python multimodal.py --configName ${my_array[$i]} | tee test.txt
        echo "finished : ${my_array[$i]} ($a/$sum)"
        result=`tail -n 3000 test.txt`
        # python sendmail.py --title "finished : ${my_array[$i]} ($a/$sum)" --content "$result"
done
# python sendmail.py --content "ALL FINISHED!" --file "YES" --title "AUTODL ALL $sum TASK FINISHED"