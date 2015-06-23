#!/bin/bash
# Script to run using GNU Parallel on amazon Cpype AMI


# XXX define python as /opt/anaconda/bin/python

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${DIR}/.."
echo $(pwd)
HOST=$(hostname)
SCRIPT=${1}
PARAMS=${*:2}
DATA_PATH="${DIR}/../data"

echo "REMOTE ${HOST}: python ${SCRIPT} ${PARAMS[@]}"

if [ "$SCRIPT" == "clean" ]; then
  echo "Cleaning up ${HOST}  #################################################"
	# rm -rf $DATA_PATH
	# rm -rf $RESULTS_PATH
elif [ "$SCRIPT" == "setup" ]; then
  echo "Setting-up ${HOST}  ##################################################"
  # update libraries
  cd ~/mne-python
  git pull git://github.com/mne-tools/mne-python
	cd ~/meeg-preprocessing
  git pull https://github.com/dengemann/meeg-preprocessing
	cd ~/meg_perceptual_decision_symbols
  git pull https://github.com/kingjr/Paris_orientation-decoding clean_up
	cd ~/gat
  git pull https://github.com/kingjr/gat

elif [ "$SCRIPT" == "upload" ]; then
  echo "Manual upload ########################################################"
	cd "${DIR}/../"
	python -c "from scripts.transfer_data import upload_all(); upload_all();" --pyscript=$SCRIPT ${PARAMS[@]}
else
	cd "${DIR}/../"

	echo "Downloading data #####################################################"
	echo "python -c \"from scripts.transfer_data import download_all; download_all()\" --pyscript=$SCRIPT ${PARAMS[@]}"
	python -c "from scripts.transfer_data import download_all; download_all()" --pyscript=$SCRIPT ${PARAMS[@]}

	echo "Running ${SCRIPT} ####################################################"
	echo python $SCRIPT --pyscript=$SCRIPT ${PARAMS[@]}
  python $SCRIPT --pyscript=$SCRIPT ${PARAMS[@]}

  # XXX concatenate log

	echo "Uploading data #######################################################"
	echo "python -c \"from scripts.transfer_data import upload_all; upload_all()\" --pyscript=$SCRIPT ${PARAMS[@]}"
	python -c "from scripts.transfer_data import upload_all; upload_all();" --pyscript=$SCRIPT ${PARAMS[@]}
fi
