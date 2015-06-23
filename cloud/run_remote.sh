#!/bin/bash
# Script to run using GNU Parallel on amazon Cpype AMI


# XXX define python as /opt/anaconda/bin/python
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${DIR}/.."
echo $(pwd)
echo $(which python)

HOST=$(hostname)
SCRIPT=${1}
PARAMS=${*:2}

echo "REMOTE ${HOST}: python ${SCRIPT} ${PARAMS[@]}"

cd ~/Paris_orientation-decoding
echo "Downloading data"
python -c "from scripts.transfer_data import download_all; download_all();" --pyscript=$SCRIPT ${PARAMS[@]}

echo "Running ${SCRIPT}"
python $SCRIPT --pyscript=$SCRIPT ${PARAMS[@]}

# XXX concatenate log

echo "Uploading data"
python -c "from scripts.transfer_data import upload_all; upload_all();" --pyscript=$SCRIPT ${PARAMS[@]}

echo 'All good!'
