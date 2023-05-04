import os
import os.path as osp
import yaml
import shutil
import torch
import warnings

from loss import PerceptualLoss
warnings.simplefilter('ignore')
import wandb
from munch import Munch
from tqdm import tqdm
from dataset import build_dataloader
from model import Discriminator,UNet


def train(config_path, num_worker):
    
    # Configs reader
    config = yaml.safe_load(open(config_path))
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    training_parameter = Munch(config.get("training_parameter"))
    ### Get configuration
    batch_size = training_parameter['batch_size']
    device = config.get('device', 'cuda:0')
    epochs = training_parameter['epochs']
    save_freq = config.get('save_freq', 20)
    dataset_configuration = config.get('dataset_configuration', None)
    training_set_path = dataset_configuration["training_set_path"]
    validation_set_path = dataset_configuration["validation_set_path"]
    lr = training_parameter['learning_rate']
    ###
    
    # load dataloader 
    train_dataloader = build_dataloader(training_set_path,dataset_configuration,
                                        batch_size=batch_size,
                                        num_workers=num_worker,
                                        device=device)
    
    val_dataloader = build_dataloader(validation_set_path,dataset_configuration,
                                        batch_size=batch_size,
                                        num_workers=num_worker,
                                        device=device,
                                        validation=True)
    
    # Generator Definition
    generators = {
        "n_to_e": UNet(),
        "e_to_n": UNet()
    }
    _ = [generator.to(device) for generator in generators.values()]
    
    gen_optimizers = {
        "n_to_e": torch.optim.Adam(generators["n_to_e"].parameters(), lr=0.0002, weight_decay=1e-4, betas=(0.5, 0.999)),
        "e_to_n": torch.optim.Adam(generators["e_to_n"].parameters(), lr=0.0002, weight_decay=1e-4, betas=(0.5, 0.999)),
    }

    # ADV Direct Discriminator Definition
    discriminators = {
        "neutral": Discriminator(),
        "emotional": Discriminator()
    }
    _ = [discriminator.to(device) for discriminator in discriminators.values()]

    dis_optimizers = {
        "neutral": torch.optim.Adam(discriminators["neutral"].parameters(), lr=0.0001),
        "emotional": torch.optim.Adam(discriminators["emotional"].parameters(), lr=0.0001),
    }
    
    start_epoch = 0
    best_val_loss = float('+inf')
    # # Load checkpoint, if exists
    if os.path.exists(osp.join(log_dir, 'backup.pth')):
        print("Loading checkpoint from {}".format(osp.join(log_dir, 'backup.pth')))
        checkpoint = torch.load(osp.join(log_dir, 'backup.pth'), map_location=device) # Fix from https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
        best_val_loss = checkpoint["loss"]
        start_epoch = checkpoint["reached_epoch"]
        for key, value in checkpoint["generators"].items():
            generators[key].load_state_dict(value) 
        for key, value in checkpoint["discriminators"].items():
            discriminators[key].load_state_dict(value)
        for key, value in checkpoint["gen_optimizers"].items():
            gen_optimizers[key].load_state_dict(value)
        for key, value in checkpoint["dis_optimizers"].items():
            dis_optimizers[key].load_state_dict(value)
        
    make_scheduler = lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.0,
        div_factor=1,
        final_div_factor=1)
    schedulers = {
        "n_to_e": make_scheduler(gen_optimizers["n_to_e"]),
        "e_to_n": make_scheduler(gen_optimizers["e_to_n"]),
    }
          
            
    for epoch in range(start_epoch, epochs):
        loss_training = {}
        loss_validation = {}
            
        # Train
        for _, batch in enumerate(tqdm(train_dataloader, desc="[train]"), 1):
            batch = [b.to(device) for b in batch]
            losses = train_epoch(batch, generators, discriminators, gen_optimizers, dis_optimizers, device, epoch)
            for key, value in losses.items():
                if key not in loss_training.keys():
                    loss_training[key] = value
                else: loss_training[key] += value
            _ = [scheduler.step() for scheduler in schedulers.values()]
            
        # Validation    
        for _, batch in enumerate(tqdm(val_dataloader, desc="[validation]"), 1):
            batch = [b.to(device) for b in batch]
            losses = eval_epoch(batch, generators, discriminators, device, epoch)
            for key, value in losses.items():
                if key not in loss_validation.keys():
                    loss_validation[key] = value
                else: loss_validation[key] += value
        
        for key, value in loss_training.items():
            loss_training[key] = value/len(train_dataloader)
        
        for key, value in loss_validation.items():
            loss_validation[key] = value/len(val_dataloader)
        
        wandb.log({"loss_training": loss_training, "loss_validation": loss_validation})
        
        save_checkpoint(osp.join(log_dir, f'backup.pth'), generators, discriminators, gen_optimizers, dis_optimizers, epoch, best_val_loss)
        if (epoch % save_freq) == 0:
            save_checkpoint(osp.join(log_dir, f'{epoch}.pth'), generators, discriminators, gen_optimizers, dis_optimizers, epoch, best_val_loss)
        
def train_epoch(batch, generators, discriminators, gen_optimizers, dis_optimizers, device, epoch) -> float:
    """Function that perform a training step on the given batch

    Args:
        batch (List[__get_item__]): Sample batch.
        model (torch.nn.Module): Model.
        loss (torch.nn.SmoothL1Loss): Loss associated.
        optimizer (torch.optim.Optimizer): Optimizer associated.

    Returns:
        float: Loss value
    """   
    _ = [generator.train() for generator in generators.values()]
    _ = [discriminator.train() for discriminator in discriminators.values()]

    real_neutral, real_emotional, real_neutral_padding_mask, real_emotional_padding_mask = batch
    
    real_neutral_padding_mask = real_neutral_padding_mask.unsqueeze(1).repeat(1, real_neutral.size(2), 1)
    real_emotional_padding_mask = real_emotional_padding_mask.unsqueeze(1).repeat(1, real_emotional.size(2), 1)
        
    lambda_lcyc = 10
    lambda_lid = 3
    loss_collector = {}
        
    generated_emotional = generators["n_to_e"](real_neutral)
    generated_neutral = generators["e_to_n"](real_emotional)
    
    
    # Direct Discriminator ############################################## 
    _ = [dis_optimizer.zero_grad() for dis_optimizer in dis_optimizers.values()]
    
        # Train discriminator Neutral
    pred_real_n = discriminators["neutral"](real_neutral)
    loss_d_real_n = torch.functional.F.binary_cross_entropy(pred_real_n, torch.ones_like(pred_real_n).to(pred_real_n.device))
    pred_fake_n = discriminators["neutral"](generated_neutral.detach())
    loss_d_fake_n = torch.functional.F.binary_cross_entropy(pred_fake_n, torch.zeros_like(pred_fake_n).to(pred_fake_n.device))
    loss_d_n = (loss_d_real_n + loss_d_fake_n) / 2.0
    loss_d_n.backward()
    dis_optimizers["neutral"].step()
    loss_collector["loss_d_n"] = loss_d_n.item()
    
        # Train discriminator Emotional
    pred_real_e = discriminators["emotional"](real_emotional)
    loss_d_real_e = torch.functional.F.binary_cross_entropy(pred_real_e, torch.ones_like(pred_real_e))
    pred_fake_e = discriminators["emotional"](generated_emotional.detach())
    loss_d_fake_e = torch.functional.F.binary_cross_entropy(pred_fake_e, torch.zeros_like(pred_fake_e))
    loss_d_e = (loss_d_real_e + loss_d_fake_e) / 2.0
    loss_d_e.backward()
    dis_optimizers["emotional"].step()
    loss_collector["loss_d_e"] = loss_d_e.item()
    
    # Generator ###################################################
    _ = [gen_optimizer.zero_grad() for gen_optimizer in gen_optimizers.values()]
    
    # Train generator N2E
    pred_fake_e = discriminators["emotional"](generated_emotional)
    loss_g_adv_N2E = torch.functional.F.binary_cross_entropy(pred_fake_e, torch.ones_like(pred_fake_e))
    loss_g_cycle_N2E_wo_mask = torch.functional.F.l1_loss(generators["e_to_n"](generated_emotional), real_neutral, reduction="none")
    loss_g_identity_N2E_wo_mask = torch.functional.F.l1_loss(generators["n_to_e"](real_emotional), real_emotional, reduction="none")
                
        # Evaluate loss with mask
    loss_g_cycle_N2E_w_mask = loss_g_cycle_N2E_wo_mask[:,0,:,:].where(real_neutral_padding_mask, torch.tensor(0.0).to(real_neutral_padding_mask.device))
    loss_g_cycle_N2E = loss_g_cycle_N2E_w_mask.sum() / real_neutral_padding_mask.sum()

    loss_g_identity_N2E_w_mask = loss_g_identity_N2E_wo_mask[:,0,:,:].where(real_emotional_padding_mask, torch.tensor(0.0).to(real_emotional_padding_mask.device))
    loss_g_identity_N2E = loss_g_identity_N2E_w_mask.sum() / real_emotional_padding_mask.sum()
        
    loss_g_N2E = loss_g_adv_N2E + lambda_lcyc * loss_g_cycle_N2E + lambda_lid * loss_g_identity_N2E 
    loss_g_N2E.backward()
    loss_collector["loss_g_adv_N2E"] = loss_g_adv_N2E.item()
    loss_collector["loss_g_cycle_N2E"] = lambda_lcyc * loss_g_cycle_N2E.item()
    loss_collector["loss_g_identity_N2E"] = lambda_lid * loss_g_identity_N2E.item()
    
    # Train generator E2N
    pred_fake_n = discriminators["neutral"](generated_neutral)
    loss_g_adv_E2N = torch.functional.F.binary_cross_entropy(pred_fake_n, torch.ones_like(pred_fake_n))
    loss_g_cycle_E2N_wo_mask = torch.functional.F.l1_loss(generators["n_to_e"](generated_neutral), real_emotional, reduction="none")
    loss_g_identity_E2N_wo_mask = torch.functional.F.l1_loss(generators["e_to_n"](real_neutral), real_neutral, reduction="none")
                                
        # Evaluate loss with mask
    loss_g_cycle_E2N_w_mask = loss_g_cycle_E2N_wo_mask[:,0,:,:].where(real_emotional_padding_mask, torch.tensor(0.0).to(real_emotional_padding_mask.device))
    loss_g_cycle_E2N = loss_g_cycle_E2N_w_mask.sum() / real_emotional_padding_mask.sum()
    
    loss_g_identity_E2N_w_mask = loss_g_identity_E2N_wo_mask[:,0,:,:].where(real_neutral_padding_mask, torch.tensor(0.0).to(real_neutral_padding_mask.device))
    loss_g_identity_E2N = loss_g_identity_E2N_w_mask.sum() / real_neutral_padding_mask.sum()
        
    loss_g_E2N = loss_g_adv_E2N + lambda_lcyc * loss_g_cycle_E2N + lambda_lid * loss_g_identity_E2N
    loss_g_E2N.backward()
    loss_collector["loss_g_adv_E2N"] = loss_g_adv_E2N.item()
    loss_collector["loss_g_cycle_E2N"] = lambda_lcyc * loss_g_cycle_E2N.item()
    loss_collector["loss_g_identity_E2N"] = lambda_lid * loss_g_identity_E2N.item()
  
        # Step
    _ = [gen_optimizer.step() for gen_optimizer in gen_optimizers.values()]
    
    return loss_collector
    
def eval_epoch(batch, generators, discriminators, device, epoch) -> float:
    """Function that perform an evaluation step on the given batch 

    Args:
        batch (List[__get_item__]): Sample batch
        model (torch.nn.Module): Model
        loss (torch.nn): Loss

    Returns:
        float: Loss value
    """    
    _ = [generator.eval() for generator in generators.values()]
    _ = [discriminator.eval() for discriminator in discriminators.values()]
    
    real_neutral, real_emotional, real_neutral_padding_mask, real_emotional_padding_mask = batch
    
    real_neutral_padding_mask = real_neutral_padding_mask.unsqueeze(1).repeat(1, real_neutral.size(2), 1)
    real_emotional_padding_mask = real_emotional_padding_mask.unsqueeze(1).repeat(1, real_emotional.size(2), 1)
    
    loss_collector = {}
    lambda_lcyc = 10
    lambda_lid = 3
    
    with torch.no_grad():
        generated_emotional = generators["n_to_e"](real_neutral)
        generated_neutral = generators["e_to_n"](real_emotional)
        
        # Direct Discriminator ############################################## 
            # Train discriminator Neutral
        pred_real_n = discriminators["neutral"](real_neutral)
        loss_d_real_n = torch.functional.F.binary_cross_entropy(pred_real_n, torch.ones_like(pred_real_n))
        pred_fake_n = discriminators["neutral"](generated_neutral.detach())
        loss_d_fake_n = torch.functional.F.binary_cross_entropy(pred_fake_n, torch.zeros_like(pred_fake_n))
        loss_d_n = (loss_d_real_n + loss_d_fake_n) / 2.0
        loss_collector["loss_d_n"] = loss_d_n.item()
        
            # Train discriminator Emotional
        pred_real_e = discriminators["emotional"](real_emotional)
        loss_d_real_e = torch.functional.F.binary_cross_entropy(pred_real_e, torch.ones_like(pred_real_e))
        pred_fake_e = discriminators["emotional"](generated_emotional.detach())
        loss_d_fake_e = torch.functional.F.binary_cross_entropy(pred_fake_e, torch.zeros_like(pred_fake_e))
        loss_d_e = (loss_d_real_e + loss_d_fake_e) / 2.0
        loss_collector["loss_d_e"] = loss_d_e.item()
        
        # Generator ###################################################
        
        # Train generator N2E
        pred_fake_e = discriminators["emotional"](generated_emotional)
        loss_g_adv_N2E = torch.functional.F.binary_cross_entropy(pred_fake_e, torch.ones_like(pred_fake_e))
        loss_g_cycle_N2E_wo_mask = torch.functional.F.l1_loss(generators["e_to_n"](generated_emotional), real_neutral, reduction="none")
        loss_g_identity_N2E_wo_mask = torch.functional.F.l1_loss(generators["n_to_e"](real_emotional), real_emotional, reduction="none")
        
            # Evaluate loss with mask
        loss_g_cycle_N2E_w_mask = loss_g_cycle_N2E_wo_mask[:,0,:,:].where(real_neutral_padding_mask, torch.tensor(0.0).to(real_neutral_padding_mask.device))
        loss_g_cycle_N2E = loss_g_cycle_N2E_w_mask.sum() / real_neutral_padding_mask.sum()

        loss_g_identity_N2E_w_mask = loss_g_identity_N2E_wo_mask[:,0,:,:].where(real_emotional_padding_mask, torch.tensor(0.0).to(real_emotional_padding_mask.device))
        loss_g_identity_N2E = loss_g_identity_N2E_w_mask.sum() / real_emotional_padding_mask.sum()
            
        loss_collector["loss_g_adv_N2E"] = loss_g_adv_N2E.item()
        loss_collector["loss_g_cycle_N2E"] = lambda_lcyc * loss_g_cycle_N2E.item()
        loss_collector["loss_g_identity_N2E"] = lambda_lid * loss_g_identity_N2E.item()
        
        # Train generator E2N
        pred_fake_n = discriminators["neutral"](generated_neutral)
        loss_g_adv_E2N = torch.functional.F.binary_cross_entropy(pred_fake_n, torch.ones_like(pred_fake_n))
        loss_g_cycle_E2N_wo_mask = torch.functional.F.l1_loss(generators["n_to_e"](generated_neutral), real_emotional, reduction="none")
        loss_g_identity_E2N_wo_mask = torch.functional.F.l1_loss(generators["e_to_n"](real_neutral), real_neutral, reduction="none")
        
            # Evaluate loss with mask
        loss_g_cycle_E2N_w_mask = loss_g_cycle_E2N_wo_mask[:,0,:,:].where(real_emotional_padding_mask, torch.tensor(0.0).to(real_emotional_padding_mask.device))
        loss_g_cycle_E2N = loss_g_cycle_E2N_w_mask.sum() / real_emotional_padding_mask.sum()
        
        loss_g_identity_E2N_w_mask = loss_g_identity_E2N_wo_mask[:,0,:,:].where(real_neutral_padding_mask, torch.tensor(0.0).to(real_neutral_padding_mask.device))
        loss_g_identity_E2N = loss_g_identity_E2N_w_mask.sum() / real_neutral_padding_mask.sum()
            
        loss_collector["loss_g_adv_E2N"] = loss_g_adv_E2N.item()
        loss_collector["loss_g_cycle_E2N"] = lambda_lcyc * loss_g_cycle_E2N.item()
        loss_collector["loss_g_identity_E2N"] = lambda_lid * loss_g_identity_E2N.item()
        
    return loss_collector

def save_checkpoint(checkpoint_path: str, generators, discriminators, gen_optimizers, dis_optimizers, epoch: int, actual_loss: float):
    """
        Save checkpoint.
        
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            model (nn.Module): Model to be saved.
            optimizer (torch.optim.Optimizer): Optimizer to be saved.
            epoch (int): Number of training epoch reached.
            actual_loss (float): Actual loss.
            global_step (int): Step counter 
    """
    state_dict = {
        "generators": {
            "n_to_e": generators["n_to_e"].state_dict(),
            "e_to_n": generators["e_to_n"].state_dict(),
        },
        "reached_epoch": epoch,
        "loss": actual_loss,
    }
    
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    torch.save(state_dict, checkpoint_path)

