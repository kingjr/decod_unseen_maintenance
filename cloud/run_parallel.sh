#!/bin/bash
# PARAMETERS
# PATH=/home/jrking/anaconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/jrking/Programs/Matlab/bin/
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${DIR}/.."
SUBJECTS=$(cat "aws/subjects.txt")
TIME_ID=$(python aws/time_id.py)
SCRIPT="scripts/run_evoked_analysis.py"
echo $TIME_ID: $SCRIPT '<' $SUBJECTS

# SETUP NODES
Configure to run on all nodes except this
THISIP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
HEADNODE=$(hostname)
# seq 0 27 | xargs -I{} echo 1/10.0.0.{} | grep -v $THISIP > nodes.slf
echo 1/$THISIP > nodes.slf
echo '' > running.slf
# Check for running nodes
bash aws/update_nodes.sh

# NODES="2/mariam,2/athena,2/jaco,1/imen,1/sami,1/flor,1/lionel"
NODES="2/:" # Run Locally
SSHLOGINS=$(echo --sshlogin $NODES)
echo "Running setup_remote.sh on all nodes"
for node in ${NODES//,/ }; do
	echo "Running on ${node}"
	parallel --ungroup --tag -S $node --transfer --cleanup "bash {1} setup" ::: "aws/run_remote.sh"
	if [ $? -ne 0 ]; then
		echo "An error ocurred while setting up ${node}"
		exit
	fi
done


# CLEAN-UP
rm -f joblog.log  # FIXME incorrect folder?
rm -f output.log
rm -f error.log

# COMPUTE
parallel --joblog job-pipeline.log --resume --resume-failed --tag --delay 1 $SSHLOGINS "bash aws/run_remote.sh ${SCRIPT} --subject={1} --time_id=${TIME_ID} --run=1" ::: $SUBJECTS > output.log 2> error.log

# log concatenation
# cleanup
# shutdown
# force shutdown
