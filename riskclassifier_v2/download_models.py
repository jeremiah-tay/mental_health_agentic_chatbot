import os, zipfile, gdown


# gdrive link for finetuned models: https://drive.google.com/file/d/1YhGQ9TQNs3pbJkEjzxcUl2K16XS4W1MF/view?usp=sharing
# gdrive file id for finetuned and trained model: 1YhGQ9TQNs3pbJkEjzxcUl2K16XS4W1MF/
FILE_ID = "1YhGQ9TQNs3pbJkEjzxcUl2K16XS4W1MF"
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
