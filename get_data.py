import kagglehub
import shutil
import os

def download_and_move():
    print("🌐 Downloading latest Mandi dataset from Kaggle...")
    # This downloads to a temporary cache folder
    path = kagglehub.dataset_download("arjunyadav99/indian-agricultural-mandi-prices-20232025")
    
    # Find the CSV file in the downloaded path
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        print("❌ No CSV file found in the download.")
        return

    source_file = os.path.join(path, files[0])
    target_dir = 'data'
    target_file = os.path.join(target_dir, 'Agriculture_price_dataset.csv')

    # Create data directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move and rename the file to our project folder
    shutil.copy(source_file, target_file)
    print(f"✅ Success! Data moved to: {target_file}")

if __name__ == "__main__":
    download_and_move()