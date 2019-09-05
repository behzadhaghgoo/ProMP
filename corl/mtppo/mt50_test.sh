for f in $1/itr_*.pkl
do
    echo $f, $2
    PYTHONPATH=/root/code/ProMP:/root/code/ProMP/metaworld:/root/code/ProMP/corl/baby python /root/code/ProMP/corl/mtppo/eval_mt50.py 0 --pkl $f --algo $2
done
