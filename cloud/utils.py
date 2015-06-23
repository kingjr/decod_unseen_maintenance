import warnings
import os
import os.path as op
import boto
import dropbox
from os.path import expanduser

credentials_dir = op.join(expanduser("~"), '.credentials')
with open('%s/boto.cfg' % credentials_dir, 'rb') as f:
    auth = f.read()
    auth = auth.split('\n')
    AWSAccessKeyId = auth[0].split('AWSAccessKeyId=')[1]
    AWSSecretKey = auth[1].split('AWSSecretKey=')[1]
boto_client = boto.connect_s3(
    AWSAccessKeyId, AWSSecretKey,
    calling_format='boto.s3.connection.OrdinaryCallingFormat').get_bucket(
        'meg.niccolo')
print('connected to S3')
with open('%s/dropbox.pem' % credentials_dir, 'rb') as f:
    auth = f.read()
dropbox_client = dropbox.client.DropboxClient(auth.split('\n')[0])
print('connected to dropbox')


def download(server, f_server, f_client, overwrite=False):
    # Deal with inexistent directories
    this_path = '/'.join(f_client.split('/')[:-1])
    if not op.exists(this_path):
        os.makedirs(this_path)

    if server == 'dropbox':
        f, metadata = dropbox_client.get_file_and_metadata(f_server)
        if op.exists(f_client):
            warnings.warn('%s already exists.' % f_client)
            if not overwrite:
                warnings.warn('%s was not overwritten.' % f_client)
                return
        out = open(f_client, 'wb')
        out.write(f.read())
        out.close()
    elif server == 's3':
        key = boto_client.get_key(f_server)
        if key is None:
            raise RuntimeError('%s is not online.' % f_client)
        # Check for overwriting
        if op.exists(f_client):
            warnings.warn('%s already exists.' % f_client)
            if not overwrite:
                warnings.warn('%s was not overwritten.' % f_client)
                return
        key.get_contents_to_filename(f_client)
    else:
        raise RuntimeError('unknown server %s' % server)
    print('downloaded from %s: %s > %s' % (server, f_server, f_client))


def upload(server, f_client, f_server, overwrite=False):
    if not op.exists(f_client):
        raise RuntimeError('%s does not exist.' % f_client)
    if server == 'dropbox':
        f = open(f_client, 'rb')
        # FIXME overwrite + check error
        response = dropbox_client.put_file(f_server, f)
    elif server == 's3':
        key = boto_client.get_key(f_server)
        if key is None:
            key = boto_client.new_key(f_server)
        elif not overwrite:
            raise RuntimeError('%s already exists online. '
                               'Set overwrite=True.' % f_server)
        key.set_contents_from_filename(f_server)
    else:
        raise RuntimeError('unknown server %s' % server)
    print('uploaded to %s > %s: %s' % (server, f_server, f_client))


def upload_all(file_list, overwrite=False):

    with open(file_list, 'rb') as f:
        file_list = f.read().split('\n')

    for fname in file_list:
        upload('s3', fname, fname, overwrite)
