from pathlib import Path
import numpy as np


# saves 2 .npz files
#                    data shape                          label shape
#                    ________________________________________________
# train file shape: (num_train_exp x num samples x 6), num_samples x 1
# test file  shape: (num_test_exp  x num samples x 6), num_samples x 1
def split_data(split, save_dir=None):
    import numpy as np
    
    num_experiments = 61
    train_users = np.ceil(num_experiments * split)
        
    # file paths to data
    acc_path  = 'data/acc/'
    gyro_path = 'data/gyro/'
    label_path = 'data/labels.txt'    
    
    # Path objects to iterate through directory
    acc_dir  = Path(acc_path)
    gyro_dir = Path(gyro_path)
    
    # list of experiments
    acc_exps  = [exp for exp in acc_dir.iterdir() ]
    gyro_exps = [exp for exp in gyro_dir.iterdir()]
    
    # labels
    labels = np.loadtxt(label_path)
      
    train_data = []
    test_data  = []
    train_labels = []
    test_labels  = []
    for i, row in enumerate(labels):

        print('{} of {}'.format(i+1, 1214), end='\r')
        
        # experiments start from 1, arrays are 0-indexed
        exp_id = int(row[0])-1       
        label_id = int(row[2])
        start = int(row[3])
        end   = int(row[4])
        
        # file paths
        a_file = acc_exps[exp_id]
        g_file = gyro_exps[exp_id]
        
        # load data in these files
        acc_exp  = np.loadtxt(a_file)
        gyro_exp = np.loadtxt(g_file)
        
        # extract data for current activity
        a = acc_exp[start:end, :]
        g = gyro_exp[start:end, :]
        
        # combine acc and gyro data into one array
        data = np.append(a, g, axis=1)

        # append data and labels to training or testing set
        if exp_id <= train_users:
            train_data.append(data)
            train_labels.append(label_id)
        else:
            test_data.append(data)
            test_labels.append(label_id)
            
        if save_dir is None:
            save_dir = 'data/combined/'
        
    print('data loaded!')

    np.savez(save_dir + 'har_train_data' + str(int(split*100)), data=train_data, labels=train_labels)
    np.savez(save_dir + 'har_test_data'  + str(int(split*100)), data=test_data,  labels=test_labels )

    print('data saved!')
        
    return train_data, train_labels, test_data, test_labels




# this function loads existining training and testing data 
def load_data(split=None, data_dir=None):
    # assume 70/30% data split, to compare with other literature
    if split is None:
        split = 0.7
        
    # determine file path
    if data_dir is None:
        data_dir = 'data/combined/'
        
    append = str(int(split*100)) + '.npz'
    
    test_file  = data_dir + 'har_test_data'  + append
    train_file = data_dir + 'har_train_data' + append
    
    # open files
    test  = np.load(test_file)
    train = np.load(train_file)
    
    # extract data
    train_data   = train['data']
    train_labels = train['labels']
    
    test_data   = test['data']
    test_labels = test['labels']
    
    return train_data, train_labels, test_data, test_labels



# This function will pad existing data to the length of the longest sample
def pad_data(data_dir=None, split=None):
    
    if data_dir is None:
        data_dir = 'data/combined/'
    if split is None:
        split = 0.7
    append = str(int(split*100)) + '_padded.npz'
    
    test_file  = data_dir + 'har_test_data'  + append
    train_file = data_dir + 'har_train_data' + append
    
    
    
    # load data to be zero-padded
    xtr, ytr, xts, yts = load_data(split=split, data_dir=data_dir)
       
    # determine the sample with the greatest length
    m_tr = max(xtr, key=len)
    m_ts = max(xts, key=len)
    
    if len(m_tr) > len(m_ts):
        padlength = len(m_tr)
    else:
        padlength = len(m_ts)
        
    # pad data to length of longest sample
    xtr_p = [ np.pad(x, ((0,padlength-len(x)),(0,0)), mode='constant', constant_values=0) for x in xtr ]
    
    xts_p = [ np.pad(x, ((0,padlength-len(x)),(0,0)), mode='constant', constant_values=0) for x in xts ]

    # save padded data
    print('saving padded data!')
    np.savez(train_file, data=xtr_p, labels=ytr)
    np.savez(test_file,  data=xts_p, labels=yts)
    print('padded data saved!')
    
    debug=False
    if debug:
        print('desired padlength: {}'.format(padlength))

        print('\ndata padded')
        print('length max xtr     {}'.format(len(max(xtr_p, key=len))))
        print('length max xts     {}'.format(len(max(xts_p, key=len))))
        print('length min xtr     {}'.format(len(min(xtr_p, key=len))))
        print('length min xts     {}'.format(len(min(xts_p, key=len))))


        print()
        print('data')
        print('length max xtr     {}'.format(len(max(xtr, key=len))))
        print('length max xts     {}'.format(len(max(xts, key=len))))
        print('length min xtr     {}'.format(len(min(xtr, key=len))))
        print('length min xts     {}'.format(len(min(xts, key=len))))
        print() 

        print()
        print('length xtr         {}'.format(len(xtr)))
        print('length xtr[0]      {}'.format(len(xtr[0])))
        print('length xtr[0][1]   {}'.format(len(xtr[0][1])))

        print()
        print('length xtr         {}'.format(len(xts)))
        print('length xtr[0]      {}'.format(len(xts[0])))
        print('length xtr[0][1]   {}'.format(len(xts[0][1]))) 
    
    












































