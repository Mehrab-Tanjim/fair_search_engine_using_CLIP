#%%
from argparse import ArgumentParser
from msilib import text
import os
from search_engine.search_engine import SearchEngine
from tqdm import tqdm 
from dataset.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from utils.utils import *
import yaml
import clip
#%%

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help="load images in batches with this batch size")
    parser.add_argument('--clip_model', type=str, default="ViT-B/32", help="different version of clip models")
    args = parser.parse_args()
    
    config_file = yaml.safe_load(open('config.yaml', 'r'))
    image_files = [os.path.join(config_file['image_folder'], image) for image in os.listdir(path=config_file['image_folder'])]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device)

    image_dataset = ImageDataset(image_files, config_file['label_path'], preprocess=preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False) 


    search_query = "face with eyeglasses"
    demography_one = "male" #TODO: ["male", "men",....]
    demography_two = "female" #TODO: ["female", "women",....]

    text_features = zeroshot_classifier([search_query, demography_one, demography_two], imagenet_templates, model)
    text_features = text_features.to(device)
    # text = clip.tokenize().to(device)
    # text_features = model.encode_text(text)

    with torch.no_grad():
        
        for i, image_batch in tqdm(enumerate(image_dataloader), total=len(image_dataset)//args.batch_size):
            
            images = image_batch['images'].to(device)
            image_paths = image_batch['image_paths']
            labels = image_batch['labels']

            image_features = model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            # logits_per_image, _ = model(images, text)
            logits_per_image = 100 * image_features @ text_features #dim x dim x dim x num_classes

            logits_per_image = logits_per_image.detach().cpu().numpy()

            relevances_scores = logits_per_image[:, 0]
            demography_scores = logits_per_image[:, 1:] # batch_size x # of demographies

            break

    # results before search
    image_paths = np.array(image_paths)

    image_grid = get_image_grid(image_paths)
    gen_plot(image_grid, title='visualization of images before search', size=10, one_channel=False, show=False)

    # making the search engine
    search_engine = SearchEngine(relevance_scores=relevances_scores, demography_scores=demography_scores)

    # base search results on relevance
    base_search_indices = search_engine.base_engine()

    image_grid = get_image_grid(image_paths[base_search_indices])
    gen_plot(image_grid, title='visualization of images after base search', size=10, one_channel=False, show=False)

    print("Diversity scores", np.std(demography_scores[base_search_indices], axis=0))

    # fmmr search results 
    fmmr_search_indices = search_engine.fmmr(list(range(relevances_scores.shape[0])), k=32, lamda=0.5)

    image_grid = get_image_grid(image_paths[fmmr_search_indices])
    gen_plot(image_grid, title='visualization of images after fmmr search', size=10, one_channel=False, show=False)

    print("Diversity scores", np.std(demography_scores[fmmr_search_indices], axis=0))
    # 1 1 1 0 0 0 -> 0 > 1.


# %%
