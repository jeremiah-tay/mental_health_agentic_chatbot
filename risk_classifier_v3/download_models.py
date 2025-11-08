import os, zipfile, gdown

FILE_ID = "1r9Bjyj-3K4XWIb5VVW8HSgiF9Z1ghEUV"
OUTPUT_ZIP = "roberta-base.zip"
DEST_DIR = "saved_models"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_and_extract():
    #get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    #set paths relative to script directory
    output_zip_path = os.path.join(script_dir, OUTPUT_ZIP)
    dest_dir_path = os.path.join(script_dir, DEST_DIR)
    
    #create saved_models directory if it doesn't exist
    os.makedirs(dest_dir_path, exist_ok=True)
    
    print(f"downloading roberta-base model from google drive...")
    gdown.download(URL, output_zip_path, quiet=False)

    print("extracting model...")
    with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir_path)
    os.remove(output_zip_path)

    print(f"model ready in: {dest_dir_path}")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
    download_and_extract()
