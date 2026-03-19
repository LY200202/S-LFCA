import time
import torch
from tqdm import tqdm
from torch import amp
import torch.nn.functional as F


def train(train_config, model, dataloader, loss_function_1, loss_function_2, optimizer, epoch, record_losses, scheduler=None, scaler=None):

    # set model train mode
    model.train()

    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    #for query, reference, ids in bar:
    for data in bar:
        data1, data2 = data
        query, labels1 = data1
        reference, labels2 = data2

        if scaler:
            with amp.autocast('cuda'):
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)
                features1_global, features1_local = features1
                features2_global, features2_local = features2
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss1 = loss_function_1(features1_global, features2_global, model.module.logit_scale1.exp(), model.module.logit_scale2.exp())
                    loss2 = (loss_function_2(features1_local, model.module.logit_scale3.exp()) + loss_function_2(features2_local, model.module.logit_scale3.exp())) / 2
                else:
                    loss1 = loss_function_1(features1_global, features2_global, model.logit_scale1.exp(), model.logit_scale2.exp())
                    loss2 = (loss_function_2(features1_local, model.logit_scale3.exp()) + loss_function_2(features2_local, model.logit_scale3.exp())) / 2
                loss = 0.8 * loss1 + 0.2 * loss2
                record_losses.update(epoch, loss.item())

            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
            
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            features1_global, features1_local = features1
            features2_global, features2_local = features2
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss1 = loss_function_1(features1_global, features2_global, model.module.logit_scale1.exp(), model.module.logit_scale2.exp())
                loss2 = (loss_function_2(features1_local, model.module.logit_scale3.exp()) + loss_function_2(features2_local, model.module.logit_scale3.exp())) / 2
            else:
                loss1 = loss_function_1(features1_global, features2_global, model.logit_scale1.exp(), model.logit_scale2.exp())
                loss2 = (loss_function_2(features1_local, model.logit_scale3.exp()) + loss_function_2(features2_local, model.logit_scale3.exp())) / 2
            loss = 0.8 * loss1 + 0.2 * loss2
            record_losses.update(epoch, loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(record_losses.global_avg_loss),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return record_losses.global_avg_loss


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with amp.autocast('cuda'):
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list