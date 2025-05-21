from tqdm import tqdm
from PIL import Image
import fitz
import os
import conf
def add_pdfs(pdf_dir):
    knowledge_base_path = conf.IMG_PAGE_DIR
    pdf_file_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    pdf_file_list = [os.path.join(pdf_dir, f) for f in pdf_file_list]

    index2img_filename = []

    for pdf_file_path in pdf_file_list:
        print(f"Processing {pdf_file_path}")
        pdf_name = os.path.basename(pdf_file_path)


        dpi = 200
        doc = fitz.open(pdf_file_path)
        
        images = []

        for (idx, page) in enumerate(tqdm(doc, desc=f"Processing {pdf_name}")):
            # with self.lock: # because we hope one 16G gpu only process one image at the same time
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)
            cache_image_path = os.path.join(knowledge_base_path, f"{pdf_name}_{idx}.jpg")
            image.save(cache_image_path, format="JPEG")
            index2img_filename.append(os.path.basename(cache_image_path))
        print(f"Finished processing {pdf_name}")
    
    with open(os.path.join(knowledge_base_path, 'index2img_filename.txt'), 'w') as f:
        f.write('\n'.join(index2img_filename))
        print(f"Saved index2img_filename to {os.path.join(knowledge_base_path, 'index2img_filename.txt')}")
    
    
if __name__ == "__main__":
    pdf_dir = conf.PDF_DIR
    add_pdfs(pdf_dir)
    print("All PDFs processed.")