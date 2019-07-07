echo $1
for i in $(seq $2 5 $3)
do
    PYTHONPATH=/root/code/ProMP:/root/code/ProMP/metaworld:/root/code/ProMP/corl/baby python /root/code/ProMP/corl/hard_test_train_envs_40_50.py 0 --pkl $1/itr_$i.pkl
done
