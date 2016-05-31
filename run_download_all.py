from config import paths
from jr.cloud import Client
client = Client('S3', bucket='meg.niccolo', multithread=False,
                client_root=paths('data_path'))
for subject in range(1, 20):
    for block in range(1, 6):
        print(subject, block)
        fname = paths('sss', subject=subject, block=block)
        client.download(fname.split('sss/')[-1], fname)
        fname = paths('behavior', subject=subject)
        client.download(fname.split('behavior/')[-1], fname)
