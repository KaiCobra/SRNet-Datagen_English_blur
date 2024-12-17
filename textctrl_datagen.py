import os
import datagen

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def builddirs(dataset_dir: str, 
              train_all_dir: str, 
              train_dirs: str,
              eval_dir: str) -> None:
    """
    Create the directories for TextCtrl dataset
     - **dataset_dir**: the root directory of the dataset
     - **train_all_dir**: dataset_fir/train
     - **train_dir**: the directories for training data, (etc. train-50k-1, train-50k-2, train-50k-3, train-50k-4)
    """

    fonts_dir = f'{dataset_dir}/fonts'
    
    makedir(dataset_dir)
    makedir(train_all_dir)
    makedir(fonts_dir)
    makedir(eval_dir)
    for dir in train_dirs:
        makedir(dir)

def main():
    
    dataset_dir = '../TextCtrl_dataset'
    train_all_dir = f'{dataset_dir}/train'
    train_dirs =[f'{train_all_dir}/train-50k-{i}' for i in range(1, 5)]
    eval_dir = f'{dataset_dir}/eval'

    builddirs(dataset_dir, train_all_dir, train_dirs, eval_dir)

    datagen.main(data_dir = eval_dir, sample_num = 1000)

    for train_dir in train_dirs:
        os.mkdir(f'{train_dir}/test')
        datagen.main(data_dir = train_dir, sample_num = 50000)
    
    
    
if __name__ == '__main__':
    main()