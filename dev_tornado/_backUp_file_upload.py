#!/usr/bin/env python3
"""
Created on Wed Aug 31 16:02:03 2022

@author: compressionist

"""

import asyncio
import mimetypes
import os
import sys
from functools import partial
from urllib.parse import quote
from uuid import uuid4
from typing import Set
from tornado import gen, httpclient
from tornado.options import define, options


class Task:

    """A command, an asynchronous task, imagine an asynchronous action
    https://stackoverflow.com/questions/57708124/correct-usage-of-asyncio-conditions-wait-for-method
    """

    def run(self):
        """To be defined in sub-classes."""

    async def start(self, condition: asyncio.Condition,
                    tasks: Set['Task']):
        """
        Start the task, calling run asynchronously.

        This method also keeps track of the running commands.

        """
        tasks.add(self)
        await self.run()
        tasks.remove(self)

        # At this point, we should ask the condition to update
        # as the number of running tasks might have reached 0.
        async with condition:
            condition.notify()


# Using HTTP POST, upload one or more files in a single multipart-form-encoded
# request.
@gen.coroutine
def multipart_producer(boundary, filenames, write):
    """
    Using HTTP POST, upload one or more files in a single multipart-form-encoded
    request
    Parameters
    ----------
    boundary : <class 'str'>
    filenames : <class 'list'>
    write : <class 'method'>
    """
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


# Using HTTP POST, upload one or more files in a single multipart-form-encoded
# request.
class Post(Task):
    """
    A subclass of a command, running a Post task:
    Using HTTP POST, upload one or more files in a single multipart-form-encoded
    request.
    """

    def __init__(self, filenames):
        self.filenames = filenames

    @gen.coroutine
    def run(self):
        print(
            f'HTTP POST - Upload {self.filenames} in a single multipart-form-encoded request:')
        client = httpclient.AsyncHTTPClient()
        boundary = uuid4().hex
        headers = {"Content-Type": "multipart/form-data; boundary=%s" % boundary}
        producer = partial(multipart_producer, boundary, self.filenames)
        response = yield client.fetch(
            "http://localhost:8888/post",
            method="POST",
            headers=headers,
            body_producer=producer,
        )
        print(response)


@gen.coroutine
def raw_producer(filename, write):
    """
    Parameters
    ----------
    filename : <class 'str'>
    write :  <class 'method'>
    """
    with open(filename, "rb") as f:
        while True:
            # 16K at a time.
            chunk = f.read(16 * 1024)
            if not chunk:
                # Complete.
                break

            yield write(chunk)


# Using HTTP PUT, upload one raw file. This is preferred for large files since
# the server can stream the data instead of buffering it entirely in memory.
class Put(Task):
    """
    A subclass of a command, running a Put task:
    Using HTTP PUT, upload one raw file. This is preferred for large files since
    the server can stream the data instead of buffering it entirely in memory.
    """

    def __init__(self, filenames):
        self.filenames = filenames

    @gen.coroutine
    def run(self):

        client = httpclient.AsyncHTTPClient()
        for filename in self.filenames:
            print(f'HTTP PUT - Upload {filename}:')
            mtype = mimetypes.guess_type(
                filename)[0] or "application/octet-stream"
            headers = {"Content-Type": mtype}
            producer = partial(raw_producer, filename)
            url_path = quote(os.path.basename(filename))
            response = yield client.fetch(
                "http://localhost:8888/%s" % url_path,
                method="PUT",
                headers=headers,
                body_producer=producer,
            )
            print(response)


async def main():
    """
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
    
    print("--GOT Filenames ---",filenames)
    put_obj = Put(filenames)
    post_obj = Post(filenames)

    method = put_obj.run() if options.put else post_obj.run()

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
