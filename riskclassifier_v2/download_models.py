import os, zipfile, gdown

# Google Drive link for fine-tuned models
# https://drive.google.com/file/d/1Hhl-sMI-usslHBP1hE8vb8KLDPyS-2ZC/view?usp=sharing
FILE_ID = "1Hhl-sMI-usslHBP1hE8vb8KLDPyS-2ZC"
OUTPUT_ZIP = "saved_models.zip"
DEST_DIR = "saved_models"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_and_extract():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading models from Google Drive...")
    gdown.download(URL, OUTPUT_ZIP, quiet=False)

    print("Extracting models into:", DEST_DIR)
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zip_ref:
        zip_ref.extractall(DEST_DIR)  # <--- FIXED: extract inside saved_models/
    os.remove(OUTPUT_ZIP)

    print(f"Models ready in: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
    download_and_extract()
