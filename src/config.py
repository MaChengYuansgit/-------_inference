

cfg = {
    ### Global Set
    "model_name": "convnext_base",  
    'GPU_ID': '0',
    "class_number": 5,
    "class_names": [], #str in list or [] for DIR label

    "random_seed":2222,
    "cfg_verbose":True,
    "num_workers":4,


    ### Train Setting
    'train_path':"/home/vipuser/train/data/trainval/split",
    'val_path':"", #if '' mean use k_flod
    'pretrained':'/home/vipuser/train/data/convnext_base_22k_1k_384.pth', #path or ''


    'try_to_train_items': 0,   # 0 means all, or run part(200 e.g.) for bug test
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "/home/vipuser/train/models/",
    'metrics': ['acc'], # default acc,  can add F1  ...
    "loss": 'CE', 

    'show_heatmap':False,
    'show_data':False,


    ### Train Hyperparameters
    "img_size": [450, 450], # [h, w]
    'learning_rate':0.0001,
    'batch_size':16,
    'epochs':100,
    'optimizer':'Adam',  #Adam  SGD AdaBelief Ranger
    'scheduler':'default-0.1-3', #default  SGDR-5-2    step-4-0.8

    'warmup_epoch':0, # 
    'weight_decay' : 0.0001,#0.0001,
    "k_flod":5,
    'val_fold':0,
    'early_stop_patient':7,

    'use_distill':0,
    'label_smooth':0,
    # 'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,


    'mixup':False,
    'cutmix':False,
    'sample_weights':None,


    ### Test
    'model_path':'/home/vipuser/train/models/exp0/best.pt',#test model

    'eval_path':"./data/test",#test with label,get eval result
    'test_path':"/home/vipuser/train/data/testB/imgs",#test without label, just show img result
    
    'TTA':False,
    'merge':False,
    'test_batch_size': 1,
    

}
