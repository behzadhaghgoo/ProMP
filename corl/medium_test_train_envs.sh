echo $1
for i in $(seq $2 5 $3)
do
    PYTHONPATH=/root/code/ProMP:/root/code/ProMP/metaworld:/root/code/ProMP/corl/baby python /root/code/ProMP/corl/medium_test_train_env.py 0 --pkl $1/itr_$i.pkl
done