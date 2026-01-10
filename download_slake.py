from huggingface_hub import hf_hub_download
import zipfile
import os

# Download the zip file
zip_path = hf_hub_download(repo_id="BoKelvin/SLAKE", filename="imgs.zip", repo_type="dataset")

# Extract it
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("slake_images")

# Example of accessing an image
# if img_name = "xmlab1/source.jpg"
# full_path = os.path.join("slake_images", "xmlab1/source.jpg")
