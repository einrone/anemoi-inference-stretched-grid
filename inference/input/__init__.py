# python standard library 
import abc
import argparse
import os 
import logging

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import cached_property
from typing import Optional

# PyPi packages 
import fsspec
import hydra 

from anemoi.datasets import open_dataset
from dotenv import load_dotenv, dotenv_values
from omegaconf import OmegaConf, DictConfig


LOGGER = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
except Exception as e:
    LOGGER.error(f"Could not import packages. Exception raised: {e}")

# Local packages

load_dotenv()
class Input(abc.ABC):
    pass 

class LocalInput:
    "should contain methods for fetching and reading files 'locally'"
    pass 

class AzureInput:
    BLOB_NAME = None 
    BLOB_CONTAINER_NAME = None
    PATH = "/home/arams/Documents/project/anemoi-inference-stretched-grid/inference/input"

    def __init__(
            self, 
            *, 
            args: Optional[Namespace] = None, 
            config: Optional[DictConfig] = None
            ) -> None:
        
        self.args = args
        self.config = config

        self.BLOB_NAME = args.blob_name
        self.BLOB_CONTAINER_NAME = args.blob_container_name
        self.lagged = 1 #args.lagged if args.lagged else 1
        CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONTAINER_STRING")

        if not CONNECTION_STRING:
            LOGGER.error("Could not fetch or find azure storage container string. Please check if it is with environment variable list")
            raise ValueError("AZURE_STORAGE_CONTAINER_STRING could not be found in the environment variable list")
        self.blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        self._check_dir

    @property
    def _check_dir(self):
        from pathlib import Path
        PATH = Path(self.PATH)
        PATH = PATH / "input"
        if not PATH.exists():
            PATH.mkdir()
        self.PATH = PATH
        LOGGER.info("Input folder path exists, will not create folder")
            


    @cached_property
    def fetch_blobs(self) -> None:
        """
            Downloads data from azure blob storage
        """
        # TODO: make this compatible with azure functions
        # and azure key vault and not use .env for keys

        # fetch blob, container name and conn str
        BLOB_CONTAINER_NAME = self.BLOB_CONTAINER_NAME
  
        try:
            # TODO : implement so it fetches latest two initial zarr states
            container_client = self.blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
            #blob_client = container_client.get_blob_client(self.blob_name)     
            blobs = [{"name" : blob.name, "container" : blob.container, "last_modified": blob.last_modified} for blob in container_client.list_blobs()]
            blobs.sort(key=lambda x : x["last_modified"], reverse=True)

            LOGGER.info("Successfully fetched blobs from azure blob storage")
            return blobs#[:self.lagged]
        
        except Exception as e:
            LOGGER.error(f"Error downloading/fetching blob. An exception occured: {e}")
            raise

    @cached_property
    def download_blob(self) -> None:
        blobs = self.fetch_blobs
        
        #for b in blobs:
        def _download_blob(b, blob_service_client):
            blob_client = blob_service_client.get_blob_client(
                    container=b['container'],
                    blob=b['name']
                )
            try:
                file_path = self.PATH / b["name"]
                file_path.parent.mkdir(parents=True, exist_ok = True)
                with open(file=file_path, mode ="wb") as currBlob:
                    download_stream=blob_client.download_blob()
                    currBlob.write(download_stream.readall())
            except Exception as e:
                LOGGER.error(f"An exception were raised: {e}")
                raise
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_download_blob, b, self.blob_service_client) for b in blobs]
            for future in as_completed(futures):
                future.result()
        return


    @cached_property
    def _read_azure_zarr(self):
        import zarr
        _blobs, container_client = self._fetch_blobs
        CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONTAINER_STRING")
        CHECK = sum([True if b["name"].endswith(".zarr") else False for b in _blobs])
        """
        if CHECK != len(_blobs):
            #LOGGER.error(f"Detected non zarr files within the blob storage. Found {len(_blobs) - CHECK} non zarr files")
            raise ValueError(f"Detected non zarr files within the blob storage. Found {len(_blobs) - CHECK} non zarr files")
        """
        #ds = open_dataset("abfs://input/"+_blobs[0]["name"])
        #print(_blobs)
        #store = zarr.ABSStore(client=container_client)
        fs = fsspec.filesystem("abfs", anon=False, connection_string = CONNECTION_STRING)
        #(dir(open_dataset))
        #exit()
        for b in _blobs:
            zarr_path =fs.get_mapper(f"abfs://{b['container']}/{b['name']}")
            print(f"abfs://{b['container']}/{b['name']}")
            store = zarr.open(fs.get_mapper(f"abfs://{b['container']}/{b['name']}"), mode = "r")
            """for z in fs.glob(zarr_path):
                with fs.open(z, "rb") as file:
                    open_dataset(file)"""
            #open_dataset(zarr_path)
        #kwargs = {"client":container_client}
        #store = open_dataset(store)
        #print(store)
        #fsspec.open(f"abfs://{self.BLOB_CONTAINER_NAME}", account_name="aifstest",account_key=CONNECTION_STRING)
        #print(fs)
        #store = [fsspec.get_mapper(f"az://{self.BLOB_CONTAINER_NAME}/{b}") for b in _blobs]
        #return store 
            
    @cached_property
    def __fetch_local_data(self) -> list:
        pass 

    def fetch(self) -> os.path:
        import time 
        if self.args.azure:
            start = time.time()
            self.download_blob
            end = time.time()
            print(f"{end - start} seconds")
        return None 
             


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test interface for input"
    )
    parser.add_argument(
        "--blob_name", help = "Name of the blob storage", type=str
    )

    parser.add_argument(
        "--blob_container_name", help = "Name of the container storage", type=str
    )
    
    parser.add_argument(
        "--azure", help="Use azure cloud services", action="store_true"
        )
    args = parser.parse_args()

    AzureInput(
        args = args
    ).fetch()