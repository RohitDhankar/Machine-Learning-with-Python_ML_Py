

from dask.distributed import Client

client = Client(n_workers=4)
print(type(client))

