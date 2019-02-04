### HW1
submit jobs:
```
    ~/Softwares/spark-2.4.0-bin-hadoop2.7/bin/spark-submit ./hw1-friendship-recommendation.py ./soc-LiveJournal1Adj.txt output.txt
```

### HW2
submit jobs:
```
    ~/Softwares/spark-2.4.0-bin-hadoop2.7/bin/spark-submit ./hw2-q2-kmeans.py ./hw2-q2-kmeans
```

run this command to merge small loss files:
```
    hadoop fs -cat ./hw2-q2-kmeans/loss.txt*/part-00000 | hadoop fs -put - loss.txt
```