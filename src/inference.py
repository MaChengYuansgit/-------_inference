import os,argparse
import random
import numpy as np        
from fire import initF, Model, Runner, Data

from config import cfg
import pandas as pd


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax




def predict(cfg):
    initF(cfg)
    model = Model(cfg)
    data = Data(cfg)

    test_loader = data.getTestDataloader()
    runner = Runner(cfg, model)
    runner.modelLoad(cfg['model_path'])

    res_dict = runner.predict(test_loader)
    print(len(res_dict))

    # to csv
    res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['defectType'])
    res_df = res_df.reset_index().rename(columns={'index':'fileName'})

    res_df = res_df[['defectType', 'fileName']]
    res_df['defectType'] = res_df['defectType'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})

    csv_names = os.listdir(cfg['test_path'].replace("imgs",'csv'))
    res_df['fileName'] = res_df['fileName'].apply(lambda x: x[:-4])

    res_df.to_csv(os.path.join(cfg['save_dir'], 'submission.csv'),
                                index=False,header=True)


def predictMerge(cfg):
    initF(cfg)

    model = Model(cfg)
    
    data = Data(cfg)

    
    test_loader = data.getTestDataloader()
    runner1 = Runner(cfg, model)
    runner1.modelLoad('output/efficientnet-b6_e17_fold0_0.93368.pth')
    print("load model1, start running.")
    res_dict1 = runner1.predictRaw(test_loader)
    print(len(res_dict1))

    test_loader = data.getTestDataloader()
    runner2 = Runner(cfg, model)
    runner2.modelLoad('output/efficientnet-b6_e18_fold1_0.94537.pth')
    print("load model2, start running.")
    res_dict2 = runner2.predictRaw(test_loader)

    test_loader = data.getTestDataloader()
    runner3 = Runner(cfg, model)
    runner3.modelLoad('output/efficientnet-b6_e14_fold2_0.91967.pth')
    print("load model3, start running.")
    res_dict3 = runner3.predictRaw(test_loader)

    test_loader = data.getTestDataloader()
    runner4 = Runner(cfg, model)
    runner4.modelLoad('output/efficientnet-b6_e18_fold3_0.92239.pth')
    print("load model4, start running.")
    res_dict4 = runner4.predictRaw(test_loader)




    res_dict = {}
    for k,v in res_dict1.items():
        v1 =np.argmax(v+res_dict2[k]+res_dict3[k]+res_dict4[k])
        res_dict[k] = v1
    
    res_list = sorted(res_dict.items(), key = lambda kv: int(kv[0].split("_")[-1].split('.')[0]))
    print(len(res_list), res_list[0])

    with open('result.csv', 'w', encoding='utf-8') as f:
        f.write('file,label\n')
        for i in range(len(res_list)):
            line = [res_list[i][0], str(res_list[i][1])]
            line = ','.join(line)
            f.write(line+"\n")


def predictTTA(cfg):

    pass

def predictMergeTTA(cfg):

    pass

def main(cfg):

    if cfg["merge"]:
        if cfg["TTA"]:
            predictMergeTTA(cfg)
        else:
            predictMerge(cfg)
    else:
        if cfg["TTA"]:
            predictTTA(cfg)
        else:
            predict(cfg)





if __name__ == '__main__':
    main(cfg)