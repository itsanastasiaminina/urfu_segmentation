import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataGenerator:
    def __init__(self, base_path, mask_path, image_path):
        self.base_path = base_path
        self.mask_folder = os.path.join(base_path, mask_path)
        self.image_folder = os.path.join(base_path, image_path)
    
    def load_files(self):
        self.mask_files = sorted(os.listdir(self.mask_folder))
        self.image_files = sorted(os.listdir(self.image_folder))
        if len(self.mask_files) != len(self.image_files):
            logging.warning("diff amount gt and img")
    
    def create_dataframe(self):
        data = []
        for mask_file, image_file in zip(self.mask_files, self.image_files):
            data.append({
                'gt_path': os.path.join(self.mask_folder, mask_file),
                'img_path': os.path.join(self.image_folder, image_file)
            })
        self.df = pd.DataFrame(data)
        self.df.reset_index(inplace=True)
    
    def clean_paths(self):
        df = self.df.copy()
        df['gt_path_clean'] = df['gt_path'].str.replace(r'\.(png|tif)$', '', regex=True)
        df['img_path_clean'] = df['img_path'].str.replace(r'\.(png|tif)$', '', regex=True)
        df['gt_path_clean'] = df['gt_path_clean'].str.replace(os.path.join(self.base_path, "gt/"), '', regex=False)
        df['img_path_clean'] = df['img_path_clean'].str.replace(os.path.join(self.base_path, "images/"), '', regex=False)
        df['same'] = (df['gt_path_clean'] == df['img_path_clean']).astype(int)
        self.result_df = df.drop(['gt_path_clean', 'img_path_clean', 'same'], axis=1)
    
    def save_all_data(self):
        os.makedirs("data", exist_ok=True)
        self.result_df.to_csv(os.path.join("data", "all_data.csv"), index=False)
        logging.info("'data/all_data.csv' saved")
    
    def split_and_save(self):
        annotator, labeled = train_test_split(self.result_df, test_size=0.2, random_state=42)
        if not os.path.exists("data"):
            os.makedirs("data")
        labeled.to_csv(os.path.join("data", "labeled_0.csv"), index=False)
        annotator.to_csv(os.path.join("data", "annotator_0.csv"), index=False)
        unlabaled_set = annotator.drop(['gt_path'], axis=1)
        unlabaled_set.to_csv(os.path.join("data", "unlabeled_0.csv"), index=False)
        logging.info("'data/labeled_0.csv', 'data/annotator.csv', 'data/unlabeled_set.csv' saved")
    
    def run(self):
        self.load_files()
        self.create_dataframe()
        self.clean_paths()
        self.save_all_data()
        self.split_and_save()

if __name__ == "__main__":
    base_path = "/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512/train"
    mask_path = "gt/"
    image_path = "images/"
    generator = DataGenerator(base_path, mask_path, image_path)
    generator.run()