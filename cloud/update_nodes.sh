#!/bin/bash
echo "UPDATE NODES ###########################################################"
rm -f tmp.slf
rm -f tmp2.slf
nice parallel --nonall -j20 -k --slf nodes.slf --tag echo >> tmp.slf
cat tmp.slf | xargs -n1 -I{} echo 1/{} >> tmp2.slf
diff tmp2.slf running.slf > tmp.tmp  # weird permission in /dev/null
if [ $? -ne 0 ]; then
  mv tmp2.slf running.slf
fi
rm -f tmp.tmp
rm -f tmp.slf
rm -f tmp2.slf
