HOST=$(hostname)
SCRIPT=${1}
PARAMS=${*:2}
echo "REMOTE ${HOST}: Setup"
PATH=/home/ubuntu/anaconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
git config --global user.email "jeanremi.king+github@gmail.com"
cd ~/mne-python
git pull git://github.com/mne-tools/mne-python
cd ~/meeg-preprocessing
git pull https://github.com/dengemann/meeg-preprocessing
cd ~/gat
git pull https://github.com/kingjr/gat
cd ~/Paris_orientation-decoding
git reset --hard
git pull https://github.com/kingjr/Paris_orientation-decoding
rm -f *.log
rm -f *.slf
