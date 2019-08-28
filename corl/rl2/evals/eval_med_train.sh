echo $1
for i in $(seq $2 1 $3)
do
    PYTHONPATH=/root/code/ProMP:/root/code/ProMP/metaworld:/root/code/ProMP/corl/baby python /root/code/ProMP/corl/rl2/evals/eval_med_train.py 0 --pkl $1/itr_$i.pkl
done
