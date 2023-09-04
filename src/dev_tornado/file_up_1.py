#!/usr/bin/env python

"""Usage: python file_upload.py [--put] file1.txt file2.png ...

Demonstrates uploading files to a server, without concurrency. It can either
POST a multipart-form-encoded request containing one or more files, or PUT a
single file without encoding.

See also file_receiver.py in this directory, a server that receives uploads.
"""

import asyncio , mimetypes , os , sys
from functools import partial
from urllib.parse import quote
from uuid import uuid4
from tornado import gen, httpclient
from tornado.options import define, options

# Using HTTP POST, upload one or more files in a single multipart-form-encoded
# request.
#@gen.coroutine -- DEPRECATED
async def multipart_producer(boundary, filenames, write):
    boundary_bytes = boundary.encode()

    for filename in filenames:
        filename_bytes = filename.encode()
        mtype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        buf = (
            (b"--%s\r\n" % boundary_bytes)
            + (
                b'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                % (filename_bytes, filename_bytes)
            )
            + (b"Content-Type: %s\r\n" % mtype.encode())
            + b"\r\n"
        )
        await write(buf)
        with open(filename, "rb") as f:
            while True:
                # 16k at a time.
                chunk = f.read(16 * 1024)
                if not chunk:
                    break
                await write(chunk)

        await write(b"\r\n")

    await write(b"--%s--\r\n" % (boundary_bytes,))

# Using HTTP PUT, upload one raw file. This is preferred for large files since
# the server can stream the data instead of buffering it entirely in memory.
#@gen.coroutine -- DEPRECATED
async def post(filenames):
    client = httpclient.AsyncHTTPClient()
    boundary = uuid4().hex
    headers = {"Content-Type": "multipart/form-data; boundary=%s" % boundary}
    producer = partial(multipart_producer, boundary, filenames)
    #response = yield client.fetch(
    response = await client.fetch(
        "http://localhost:8877/post",
        method="POST",
        headers=headers,
        body_producer=producer,
    )

    print("[INFO]--post-response--->/n",response)

#@gen.coroutine -- DEPRECATED 
async def raw_producer(filename, write):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(16 * 1024) # 16K at a time.
            if not chunk:
                print("[INFO]--No more File Chunks---")# Complete.
                break
            #yield write(chunk)
            await write(chunk)

#@gen.coroutine -- DEPRECATED 
async def put(filenames):
    client = httpclient.AsyncHTTPClient()
    for filename in filenames:
        mtype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        headers = {"Content-Type": mtype}
        print("[INFO]--put---headers--->",headers)
        producer = partial(raw_producer, filename)
        url_path = quote(os.path.basename(filename))
        #response = yield client.fetch(
        response = await client.fetch(
            "http://localhost:8877/%s" % url_path,
            method="PUT",
            headers=headers,
            body_producer=producer,
        )
        print("[INFO]--put-response--->/n",response)

if __name__ == "__main__":
    define("put", type=bool, help="Use PUT instead of POST", group="file uploader")
    # Tornado configures logging from command line opts and returns remaining args.
    filenames = options.parse_command_line()
    if not filenames:
        print("Give filenames...", file=sys.stderr)
        sys.exit(1)

    #method = put(filenames) if options.put else post(filenames)
    method = put if options.put else post
    asyncio.run(method(filenames))
    #asyncio.run(put(filenames))


"""
(dbfs_env) dhankar@dhankar-1:~/.../torn$ python file_up_1.py fl_1.log 
Traceback (most recent call last):
  File "file_up_1.py", line 106, in <module>
    asyncio.run(method(filenames))
TypeError: 'coroutine' object is not callable
sys:1: RuntimeWarning: coroutine 'post' was never awaited

"""