for f in $1/itr_*.pkl
do
    echo $f
    PYTHONPATH=/root/code/ProMP:/root/code/ProMP/metaworld:/root/code/ProMP/corl/baby python /root/code/ProMP/corl/hard_test_$2_envs.py 0 --pkl $f
done
