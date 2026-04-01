import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import shutil
from torch.amp import GradScaler
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from trainer import train
from evaluate import evaluate
from model import Model
from torch.utils.data import DataLoader
from utils import setup_system, Logger,  LossLogger
from datasets.U1652Dataset import *
from datasets.make_dataloader import *
from losses.global_infonce_loss import GlobalInfoNCE
from losses.local_infonce_loss import LocalInfoNCE
from datetime import datetime
import pytz
import argparse
from shutil import copyfile, copytree
from count_model_stats import count_model_stats


def configuration():
    parser = argparse.ArgumentParser(description="Training Configuration")

    # Model
    parser.add_argument("--model", default="dinov2_vitb14", type=str, help="Model name")
    parser.add_argument("--num_trainable_blocks", default=4, type=int, help="num_trainable_blocks")
    parser.add_argument("--img_size", default=378, type=int, help="Override model input image size")

    # Training
    parser.add_argument("--mixed_precision", default=True, type=bool, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--seed", default=1, type=int, help="Random seed for reproducibility")
    parser.add_argument("--epochs", default=120, type=int, help="Number of training epochs")
    parser.add_argument("--classes_num", default=20, type=int, help="Number of classes (P)")
    parser.add_argument("--sample_num", default=3, type=int, help="Number of samples (K)")
    parser.add_argument("--verbose", default=True, type=bool, help="Print detailed logs during training")
    parser.add_argument("--gpu_ids", default=(0,), type=int, nargs="+", help="List of GPU IDs to use for training")
    parser.add_argument("--save_weights", default=True, type=bool, help="Save model weights during training")

    # Evaluation
    parser.add_argument("--batch_size_eval", default=128, type=int, help="Batch size used during evaluation")
    parser.add_argument("--eval_every_n_epoch", default=5, type=int, help="Run evaluation every N epochs")
    parser.add_argument("--normalize_features", default=True, type=bool, help="Normalize embeddings before distance computation")
    parser.add_argument("--eval_gallery_n", default=-1, type=int, help="-1 = use all gallery images, else specify the number")

    # Optimizer
    parser.add_argument("--clip_grad", default=100., type=float, help="Clip gradient norm value. None means no clipping")
    parser.add_argument("--decay_exclue_bias", default=False, type=bool, help="Exclude bias terms from weight decay")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight_decay")
    parser.add_argument("--grad_checkpointing", default=True, type=bool, help="Enable gradient checkpointing for memory saving")

    # Loss
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing factor")

    # Learning Rate
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--logit_scale_lr", default=0.001, type=float, help="Learning rate for logit scale (contrastive models)")
    parser.add_argument("--scheduler", default="cosine", type=str, choices=["polynomial", "cosine", "constant", "none"], help="Learning rate schedule strategy")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Warmup epochs for scheduler")
    parser.add_argument("--lr_end", default=0.00001, type=float, help="Final learning rate for polynomial LR scheduler")

    # Dataset
    parser.add_argument("--dataset", default="U1652-D2S", type=str, choices=["U1652-D2S", "U1652-S2D"], help="Dataset selection")
    parser.add_argument("--data_folder", default="/mnt/data1/liyong/University-Release/", type=str, help="Dataset folder root path")

    # Data Augmentation
    parser.add_argument("--prob_flip", default=0.5, type=float, help="Probability of flipping both satellite and drone images")

    # Save Paths
    parser.add_argument("--model_path", default="./university", type=str, help="Path to save model logs or results")

    # Evaluation before training
    parser.add_argument("--zero_shot", default=False, type=bool, help="Run zero-shot evaluation before training starts")

    # Checkpoint resume
    parser.add_argument("--checkpoint_start", default=None, type=str, help="Path to checkpoint to resume training")

    # Dataloader workers
    parser.add_argument("--num_workers", default=0 if os.name == "nt" else 4, type=int, help="Number of DataLoader workers. Windows must use 0")

    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use: cuda or cpu")

    # CuDNN configs
    parser.add_argument("--cudnn_benchmark", default=True, type=bool, help="Enable CuDNN benchmark mode")
    parser.add_argument("--cudnn_deterministic", default=False, type=bool, help="Make CuDNN deterministic (may reduce speed)")

    args = parser.parse_args()
    return args


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = configuration()
config.batch_size = config.classes_num * config.sample_num
config.data_folder_train = os.path.join(config.data_folder,'train')
config.data_folder_test = os.path.join(config.data_folder,'test')

if config.dataset == 'U1652-D2S':
    config.query_folder_train = os.path.join(config.data_folder_train, "satellite")
    config.gallery_folder_train = os.path.join(config.data_folder_train, "drone")
    config.query_folder_test = os.path.join(config.data_folder_test, "query_drone")
    config.gallery_folder_test = os.path.join(config.data_folder_test, "gallery_satellite")
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = os.path.join(config.data_folder_train, "satellite")
    config.gallery_folder_train = os.path.join(config.data_folder_train, "drone")
    config.query_folder_test = os.path.join(config.data_folder_test, "query_satellite")
    config.gallery_folder_test = os.path.join(config.data_folder_test, "gallery_drone")


if __name__ == '__main__':

    # Create model save path
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz).strftime("%Y_%m_%d_%H%M")
    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   current_time)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    config.weights_save_path = os.path.join(model_path,'weights')
    if not os.path.exists(config.weights_save_path):
        os.makedirs(config.weights_save_path)

    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    sys.stdout = Logger(os.path.join(model_path, 'train_log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    ########################################
    # training details
    print("\n{}[*training details*]{}".format(30 * "-", 30 * "-"))
    print(f"epoch = {config.epochs}，"
          f"classes_num = {config.classes_num}，"
          f"sample_num = {config.sample_num}，"
          f"batch size = {config.batch_size}，"
          f"use {config.scheduler} scheduler，"
          f"lr = {config.lr}，"
          f"warmup_epochs = {config.warmup_epochs}，")


    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))
    print(f"num_trainable_blocks = {config.num_trainable_blocks}")

    model = Model(model_name=config.model, num_trainable_blocks=config.num_trainable_blocks)
    count_model_stats(model=model, input_size=(1, 3, config.img_size, config.img_size))
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    #if config.grad_checkpointing:
        #model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
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
                                                                                                                                 
    # Train
    train_dataloader, class_names, dataset_sizes = get_train_dataloader(config, data_transforms)
    print('dataset_sizes:', "(satellite:{}，drone:{})".format(dataset_sizes['satellite'], dataset_sizes['drone']))
    
    # Reference Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms)
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n)
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))


    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn_1 = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    global_info_loss = GlobalInfoNCE(loss_function=loss_fn_1,
                                     sample_num=config.sample_num)
                                  
    loss_fn_2 = torch.nn.CrossEntropyLoss()
    local_info_loss = LocalInfoNCE(loss_function=loss_fn_2)

    if config.mixed_precision:
        scaler = GradScaler('cuda', init_scale=2.**10)
    else:
        scaler = None


    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    factor_1 = round((14 / len(train_dataloader)) * 1, 2)
    factor_2 = round((14 / len(train_dataloader)) * 2, 2)
    print("\nTemperature coefficient scaling factor:")
    print("factor_1: {}  factor_2: {}".format(factor_1,factor_2))

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "norm.bias", "norm.weight"]
        exclude_params = ["logit_scale1", "logit_scale2", "logit_scale3", "featuregrouping"]

        decay_params = [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)
               and not any(ep in n for ep in exclude_params)
        ]

        no_decay_params = [
            p for n, p in param_optimizer
            if any(nd in n for nd in no_decay)
               and not any(ep in n for ep in exclude_params)
        ]

        featuregrouping_decay_params = [
            p for n, p in param_optimizer
            if "featuregrouping" in n
               and not any(nd in n for nd in no_decay)
        ]

        featuregrouping_no_decay_params = [
            p for n, p in param_optimizer
            if "featuregrouping" in n
               and any(nd in n for nd in no_decay)
        ]

        optimizer_parameters = [
            {"params": decay_params, "weight_decay": config.weight_decay, "lr": config.lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": config.lr},
            {"params": featuregrouping_decay_params, "weight_decay": config.weight_decay, "lr": config.lr * 2},
            {"params": featuregrouping_no_decay_params, "weight_decay": 0.0, "lr": config.lr * 2},
            {"params": [model.logit_scale1], "weight_decay": 0.0, "lr": config.logit_scale_lr * factor_1},
            {"params": [model.logit_scale2], "weight_decay": 0.0, "lr": config.logit_scale_lr * factor_2},
            {"params": [model.logit_scale3], "weight_decay": 0.0, "lr": config.logit_scale_lr},
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters)
    else:
        params = [
            {'params': model.model.parameters(), 'lr': config.lr},
            {'params': model.featuregrouping.parameters(), 'lr': config.lr * 2},
            {'params': [model.logit_scale1], 'lr': config.logit_scale_lr * factor_1},
            {'params': [model.logit_scale2], 'lr': config.logit_scale_lr * factor_2},
            {'params': [model.logit_scale3], 'lr': config.logit_scale_lr},
        ]
        optimizer = torch.optim.AdamW(params)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = int(len(train_dataloader) * config.epochs)
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
                

    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    loss_log_dir = os.path.join(model_path, "loss")
    record_losses = LossLogger(log_dir=loss_log_dir)
    copytree('./losses', loss_log_dir, dirs_exist_ok=True)
    copyfile('./model.py',  model_path + "/model.py")


    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function_1=global_info_loss,
                           loss_function_2=local_info_loss,
                           optimizer=optimizer,
                           epoch=epoch,
                           record_losses=record_losses,
                           scheduler=scheduler,
                           scaler=scaler)

        record_losses.end_epoch()
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]['lr']))

        for i in [1, 2, 3]:
            scale = getattr(model, f"logit_scale{i}")
            for group in optimizer.param_groups:
                if any(p is scale for p in group['params']):
                    print(f"logit_scale{i}_lr: {group['lr']:.6f}  logit_scale{i}: {scale.item()}")


        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        #if (epoch % config.eval_every_n_epoch == 0 and epoch > 80) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if config.save_weights and best_score > 0.975:
                    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                        torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(config.weights_save_path, epoch, r1_test))
                    else:
                        torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(config.weights_save_path, epoch, r1_test))
                    with open('{}/results.txt'.format(model_path), 'a') as f:
                        f.write('epoch={}, R@1={:.4f}\n'.format(epoch, r1_test*100))
                else:
                    with open('{}/results.txt'.format(model_path), 'a') as f:
                        f.write('epoch={}, R@1={:.4f}\n'.format(epoch, r1_test*100))

    # save the last weight
    if config.save_weights:
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(config.weights_save_path))
        else:
            torch.save(model.state_dict(), '{}/weights_end.pth'.format(config.weights_save_path))

    # Save the final results after training
    summary = record_losses.finalize()
    print("\nTraining complete!")
    print(f"Epoch loss log saved to: {os.path.join(record_losses.log_dir, 'loss_log.txt')}")
    print("Training summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")