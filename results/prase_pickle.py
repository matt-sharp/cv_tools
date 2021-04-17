import pickle

def prase_pickle(file_path):
    f = open(file_path,'rb')
    data = pickle.load(f)
    return data

if __name__=="__main__":
    file_path = '/home/zoucg/cv_project/s2anet/z_log/restult.pkl'
    prase_pickle(file_path)