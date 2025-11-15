import os, zipfile, gdown


# gdrive link for finetuned models: https://drive.google.com/file/d/1YhGQ9TQNs3pbJkEjzxcUl2K16XS4W1MF/view?usp=sharing
# gdrive file id for finetuned and trained model: 1YhGQ9TQNs3pbJkEjzxcUl2K16XS4W1MF/
FILE_ID = "1YhGQ9TQNs3pbJkEjzxcUl2K16XS4W1MF"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ZIP = os.path.join(SCRIPT_DIR, "saved_models.zip")
DEST_DIR = os.path.join(SCRIPT_DIR, "saved_models")
URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_and_extract():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading models from Google Drive...")
    gdown.download(URL, OUTPUT_ZIP, quiet=False)

    print("Extracting models...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zip_ref:
        # Check if zip contains a saved_models folder
        zip_contents = zip_ref.namelist()
        if any(name.startswith("saved_models/") for name in zip_contents):
            # Zip already has saved_models folder, extract to parent
            zip_ref.extractall(SCRIPT_DIR)
        else:
            # Zip contents are at root, extract to DEST_DIR
            zip_ref.extractall(DEST_DIR)
    os.remove(OUTPUT_ZIP)
    print(f"Models ready in: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
    download_and_extract()
