import os
import requests

# Predefined model_configs dictionary
model_configs = {
    "ont_cdna": {
        "config": "https://zenodo.org/records/13924394/files/wtc11_cdna.yaml",
        "model": "https://zenodo.org/records/13924394/files/cdna_wtc11_nopass_resnet50_sgd.epoch30.chkpt"
    },
    "ont_drna": {
        "config": "https://zenodo.org/records/13924394/files/gm12878_drna.yaml",
        "model": "https://zenodo.org/records/13924394/files/drna_gm12878_nopass_resnet50_sgd.epoch30.chkpt"
    },
    "pb_isoseq": {
        "config": "https://zenodo.org/records/13924394/files/hg002_isoseq.yaml",
        "model": "https://zenodo.org/records/13924394/files/hg002_baylor_isoseq_nopass_resnet50_sgd.epoch30.chkpt"
    },
    "pb_masseq": {
        "config": "https://zenodo.org/records/13924394/files/hg002_na24385_masseq.yaml",
        "model": "https://zenodo.org/records/13924394/files/hg002_na24385_mix_nopass_resnet50_sgd.epoch30.chkpt"
    }
}


def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {destination}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


def download_configs_and_models(args):
    """
    Download the configuration files and models from the URLs specified in the model_configs dictionary.
    """
    download_dir = args.download_dir

    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Iterate through each section in the model_configs and download files
    for model_type, data in model_configs.items():
        config_url = data['config']
        model_url = data['model']

        # Define destination paths
        config_destination = os.path.join(download_dir, f"{model_type}_config.yaml")
        model_destination = os.path.join(download_dir, f"{model_type}_model.chkpt")

        # Download config and model files
        download_file(config_url, config_destination)
        download_file(model_url, model_destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download configuration and model files")
    parser.add_argument('-d', '--download_dir', type=str, help='Download directory', default='models')
    args = parser.parse_args()
    download_configs_and_models(args)
