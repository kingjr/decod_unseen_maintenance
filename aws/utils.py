import os.path as op


def dropbox_client():
    import dropbox
    # read credentials
    with open('/home/jrking/credentials/dropbox.pem', 'rb') as f:
        auth = f.read()
    # connect to account
    client = dropbox.client.DropboxClient(auth.split('\n')[0])
    # print 'linked account: ', client.account_info()
    base_path = '/no_sync/niccolo_data/'
    return client, base_path


def dropbox_download(f_server, f_client=None):
    f_client = f_server if f_client is None else f_client
    client, base_path = dropbox_client()
    f_server = op.join(base_path, f_server)
    f, metadata = client.get_file_and_metadata(f_server)
    out = open(f_client, 'wb')
    out.write(f.read())
    out.close()
    print('downloadded: %s > %s' % (f_server, f_client))


def dropbox_upload(f_client, f_server=None):
    f_server = f_client if f_server is None else f_client
    client, base_path = dropbox_client()
    f = open(f_client, 'rb')
    f_server = op.join(base_path, f_server)
    response = client.put_file(op.join(base_path, f_client), f)
    print 'uploaded: ', response
    print('downloaded: %s > %s' % (f_server, f_client))
