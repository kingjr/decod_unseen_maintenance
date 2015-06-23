#!/bin/bash
# Script to run using GNU Parallel on amazon Cpype AMI


# XXX define python as /opt/anaconda/bin/python
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${DIR}/.."
echo $(pwd)
HOST=$(hostname)
SCRIPT=${1}
PARAMS=${*:2}

echo "REMOTE ${HOST}: python ${SCRIPT} ${PARAMS[@]}"

# if [ "$SCRIPT" == "setup" ]; then
#   echo "Setting-up ${HOST}  ##################################################"
#   # update libraries
# 	git config --global user.email "jeanremi.king+github@gmail.com"
#   cd ~/mne-python
#   git pull git://github.com/mne-tools/mne-python
# 	cd ~/meeg-preprocessing
#   git pull https://github.com/dengemann/meeg-preprocessing
# 	cd ~/gat
#   git pull https://github.com/kingjr/gat
#   cd ~/Paris_orientation-decoding
#   git pull https://github.com/kingjr/Paris_orientation-decoding
#
# else
#
# 	cd ~/Paris_orientation-decoding
# 	echo "Downloading data"
#   /home/ubuntu/anaconda/bin/python -c "from scripts.transfer_data import download_all; download_all();" --pyscript=$SCRIPT ${PARAMS[@]}
#
# 	echo "Running ${SCRIPT}"
#   /home/ubuntu/anaconda/bin/python python $SCRIPT --pyscript=$SCRIPT ${PARAMS[@]}
#
#   # XXX concatenate log
#
# 	echo "Uploading data"
#   /home/ubuntu/anaconda/bin/python python -c "from scripts.transfer_data import upload_all; upload_all();" --pyscript=$SCRIPT ${PARAMS[@]}
#
#   echo 'All good!'
# fi
