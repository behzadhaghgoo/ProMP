echo "$1"

for i in $(seq 0 $2)
do
    PYTHONPATH=/root/code/ProMP:/root/code/ProMP/metaworld:/root/code/ProMP/corl/baby python ./maml_reach_test.py 0 --pkl $1/itr_$i.pkl
done