import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch.utils.data import DataLoader
from datasets.make_dataloader import get_transforms
from datasets.U1652Dataset import U1652DatasetEval
from evaluate import evaluate
from model import Model
import argparse


def configuration():
    parser = argparse.ArgumentParser(description="Evaluate Configuration")

    # Model
    parser.add_argument("--model", default="dinov2_vitb14", type=str, help="Model name")
    parser.add_argument("--img_size", default=378, type=int, help="Override model input image size")
    
    # Evaluation
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used during evaluation")
    parser.add_argument("--verbose", default=True, type=bool, help="Print detailed logs during evaluation")
    parser.add_argument("--gpu_ids", default=(0,), type=int, nargs="+", help="List of GPU IDs to use for training")
    parser.add_argument("--normalize_features", default=True, type=bool, help="Normalize embeddings before distance computation")
    parser.add_argument("--eval_gallery_n", default=-1, type=int, help="-1 = use all gallery images, else specify the number")
    
    # Dataset
    parser.add_argument("--dataset", default="SUES200-D2S", type=str, choices=["SUES200-D2S", "SUES200-S2D"], help="Dataset selection")
    parser.add_argument("--data_folder", default="/mnt/data1/liyong/SUES-200/", type=str, help="Dataset folder root path")
    parser.add_argument("--height", default="150", type=str, help="150|200|250|300")
    
    # Path to the eval_weight
    parser.add_argument("--eval_weight", default="weights_e80_0.9995.pth", type=str, help="Path to the weight file for evaluation")
  
    # set num_workers to 0 if on Windows
    parser.add_argument("--num_workers", default=0 if os.name == "nt" else 4, type=int, help="Number of DataLoader workers. Windows must use 0")
    
    # train on GPU if available
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use: cuda or cpu")

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = configuration()
config.data_folder_train = os.path.join(config.data_folder,'Training', config.height)
config.data_folder_test = os.path.join(config.data_folder,'Testing', config.height)

if config.dataset == 'SUES200-D2S':
    config.query_folder_train = os.path.join(config.data_folder_train, "satellite")
    config.gallery_folder_train = os.path.join(config.data_folder_train, "drone")
    config.query_folder_test = os.path.join(config.data_folder_test, "query_drone")
    config.gallery_folder_test = os.path.join(config.data_folder_test, "gallery_satellite")
elif config.dataset == 'SUES200-S2D':
    config.query_folder_train = os.path.join(config.data_folder_train, "satellite")
    config.gallery_folder_train = os.path.join(config.data_folder_train, "drone")
    config.query_folder_test = os.path.join(config.data_folder_test, "query_satellite")
    config.gallery_folder_test = os.path.join(config.data_folder_test, "gallery_drone")


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

    model = Model(model_name=config.model)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    # load eval_weight
    if config.eval_weight is not None:
        print("Start from:", config.eval_weight)
        model_state_dict = torch.load(config.eval_weight)
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    else:
        config.device = torch.device(f"cuda:{config.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    data_transforms = get_transforms(img_size, mean=mean, std=std)
    val_transforms = data_transforms['val']
    train_sat_transforms = data_transforms['satellite']
    train_drone_transforms = data_transforms['drone']
                                                                                                                                 
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms)
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n)
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
   

    print("\n{}[{}]{}".format(30*"-", "SUES200", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
 
