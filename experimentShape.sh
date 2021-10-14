#!/bin/bash

mkdir resultShape
python experiments.py --inputFolder=testShape/ --outputFolder=resultShape --target=datasetShape.cla --query=queryShape.cla --granularity=all
 