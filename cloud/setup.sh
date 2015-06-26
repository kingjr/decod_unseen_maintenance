HOST=$(hostname)
SCRIPT=${1}
PARAMS=${*:2}
rm *.log
echo "REMOTE ${HOST}: Setup"
git config --global user.email "jeanremi.king+github@gmail.com"
cd ~/mne-python
git pull git://github.com/mne-tools/mne-python
cd ~/meeg-preprocessing
git pull https://github.com/dengemann/meeg-preprocessing
cd ~/gat
git pull https://github.com/kingjr/gat
cd ~/Paris_orientation-decoding
git pull https://github.com/kingjr/Paris_orientation-decoding
rm *.log
rm *.slf
