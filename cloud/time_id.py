import time
catime = lambda ii, jj: '-'.join([str(idx) for idx in time.localtime()[ii:jj]])
time_id = catime(0, 3) + '_' + catime(3, 6)
print time_id
