#!/bin/bash
# prevent ssh
echo 'Host 10.0.0.*
   StrictHostKeyChecking no
   UserKnownHostsFile=/dev/null' | cat - /etc/ssh/ssh_config > temp
sudo mv temp /etc/ssh/ssh_config
