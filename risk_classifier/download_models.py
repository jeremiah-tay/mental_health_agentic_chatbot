import os, zipfile, gdown

# gdrive file id for finetuned and trained model: 1fINDPOJtQwQrGsRbcT1_M_Gta0FPFotJ
FILE_ID = "1fINDPOJtQwQrGsRbcT1_M_Gta0FPFotJ" 
OUTPUT_ZIP = "saved_models.zip"
DEST_DIR = "saved_models"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_and_extract():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading models from Google Drive...")
    gdown.download(URL, OUTPUT_ZIP, quiet=False)

    print("Extracting models...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(OUTPUT_ZIP)

    print(f"Models ready in: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
    download_and_extract()
