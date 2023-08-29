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
@gen.coroutine
def multipart_producer(boundary, filenames, write):
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
        yield write(buf)
        with open(filename, "rb") as f:
            while True:
                # 16k at a time.
                chunk = f.read(16 * 1024)
                if not chunk:
                    break
                yield write(chunk)

        yield write(b"\r\n")

    yield write(b"--%s--\r\n" % (boundary_bytes,))


# Using HTTP PUT, upload one raw file. This is preferred for large files since
# the server can stream the data instead of buffering it entirely in memory.
@gen.coroutine
def post(filenames):
    client = httpclient.AsyncHTTPClient()
    boundary = uuid4().hex
    headers = {"Content-Type": "multipart/form-data; boundary=%s" % boundary}
    producer = partial(multipart_producer, boundary, filenames)
    response = yield client.fetch(
        "http://localhost:8877/post",
        method="POST",
        headers=headers,
        body_producer=producer,
    )

    print("[INFO]--post-response--->/n",response)


@gen.coroutine
def raw_producer(filename, write):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(16 * 1024) # 16K at a time.
            if not chunk:
                print("[INFO]--No more File Chunks---")# Complete.
                break
            yield write(chunk)


@gen.coroutine
def put(filenames):
    client = httpclient.AsyncHTTPClient()
    for filename in filenames:
        mtype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        headers = {"Content-Type": mtype}
        print("[INFO]--put---headers--->",headers)
        producer = partial(raw_producer, filename)
        url_path = quote(os.path.basename(filename))
        response = yield client.fetch(
            "http://localhost:8877/%s" % url_path,
            method="PUT",
            headers=headers,
            body_producer=producer,
        )
        print("[INFO]--put-response--->/n",response)




async def main():
    """
    SOURCE -- https://github.com/tornadoweb/tornado/issues/3182
    Returns
    -------
    None.

    """
    define("put", type=bool, help="Use PUT instead of POST", group="file uploader")

    # Tornado configures logging from command line opts and returns remaining args
    filenames = options.parse_command_line()
    if not filenames:
        print("Provide a list of filenames to upload.", file=sys.stderr)
        sys.exit(1)
    method = put if options.put else post

    # Main coroutine, starting tasks:
    cond = asyncio.Condition()
    tasks = set()
    tasks.add(method(filenames))
    asyncio.gather(*(task.start(cond, tasks) for task in tasks))

    # Wait for the number of tasks to reach 0
    async with cond:
        await cond.wait_for(lambda: len(tasks) == 0)
        print("ALL TASKS are COMPLETED, EXITING.")

asyncio.run(main())


# if __name__ == "__main__":
#     define("put", type=bool, help="Use PUT instead of POST", group="file uploader")

#     # Tornado configures logging from command line opts and returns remaining args.
#     filenames = options.parse_command_line()
#     if not filenames:
#         print("Give filenames...", file=sys.stderr)
#         sys.exit(1)

#     method = put if options.put else post
#     asyncio.run(method(filenames))

"""
https://github.com/tornadoweb/tornado/issues/3182

https://bugs.python.org/issue36222

Traceback (most recent call last):
  File "file_upload.py", line 110, in <module>
    asyncio.run(method(filenames))
  File "/home/dhankar/anaconda3/envs/dbfs_env/lib/python3.8/asyncio/runners.py", line 37, in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
ValueError: a coroutine was expected, got <Future pending cb=[coroutine.<locals>.wrapper.<locals>.<lambda>() at /home/dhankar/anaconda3/envs/dbfs_env/lib/python3.8/site-packages/tornado/gen.py:251]>

"""