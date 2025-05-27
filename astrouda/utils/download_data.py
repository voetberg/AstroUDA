"""
Simple script to download data from the paper

Only arg is sys.argv[1], the data directory
If not provided, it will use the current directory

Downloads both LSST and Galaxy Zoo from zenodo

These are large datasets, so it may take a while to download
"""

import os
import urllib.request
import shutil

def download_lsst(path):
    download_small = True

    download_all_link = "https://zenodo.org/api/records/5514180/files-archive"

    if download_small: 
        download_small_links = [
            f"https://zenodo.org/records/5514180/files/images_{i}_150.npy" for i in [
                "Y10_test",
                "Y1_test"]
        ]
        
        for link in download_small_links:
            name = link.split("/")[-1].split("?")[0]
            if os.path.exists(os.path.join(path, name)):
                print(f"File {name} already exists, skipping download")
            else: 
                try: 
                    urllib.request.urlretrieve(link, os.path.join(path, name))
                except Exception as e:
                    print(f"Error downloading {name}: {e}")
                    
    else: 
        name = "lsst_data.zip"
        urllib.request.urlretrieve(download_all_link, os.path.join(path, name))
        print("Unzipping lsst data...")
        shutil.unpack_archive(os.path.join(path, name), path)
        

def download_galaxy_zoo(path):
    """
    Download the galaxy zoo data from zenodo
    """

    download_link = "https://zenodo.org/api/records/7473597/files-archive"
    unzipped_names = ["decals.h5", "sdss_1.h5", "sdss_2.h5", "sdss_stripe82.h5"]

    name = "galaxy_zoo_data.zip"
    dir_contents = os.listdir(path)
    if all(name in dir_contents for name in unzipped_names):
        print("Galaxy Zoo data already exists, skipping download")
    elif any(name in dir_contents for name in unzipped_names):
        print("Galaxy Zoo data partially exists, downloading missing files")
        download_list = [name for name in unzipped_names if name not in dir_contents]
        for name in download_list:
            file_link = f"https://zenodo.org/records/7473597/files/{name}?download=1"
            urllib.request.urlretrieve(file_link, os.path.join(path, name))

    else: 
        urllib.request.urlretrieve(download_link, os.path.join(path, name))
        print("Unzipping galaxy zoo data...")
        shutil.unpack_archive(os.path.join(path, name), path)

    

def main(): 
    import sys 
    if len(sys.argv) < 2:
        path = os.getcwd()

    else:
        path = sys.argv[1]
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    print(f"Downloading data to {path}")
    print("Downloading LSST data...")
    download_lsst(path)

    # print("Downloading Galaxy Zoo data...")
    # download_galaxy_zoo(path)

    print("Done!")

if __name__ == "__main__":
    main()
