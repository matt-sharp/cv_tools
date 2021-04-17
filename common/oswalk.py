import os

def os_walk(source_dir,dest_dir= './temp11',suffex=''):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    absolute_list = []
    relative_list = []
    relative_dir = []

    for rs,ds,fs in os.walk(source_dir):
        for i in fs:
            if i.endswith(suffex):
                a_p = os.path.join(rs,i)
                r_p = a_p.split(source_dir)[-1]
                absolute_list.append(a_p)
                relative_list.append(r_p)
        for j in ds:
            a_d = os.path.join(rs,j)
            r_d = a_d.split(source_dir)[-1]
            relative_dir.append(r_d)
            
    dest_pathes = list(map(lambda x:os.path.join(dest_dir,x),relative_list))
    [os.makedirs(i) for i in dest_pathes]
    # map(os.makedirs,dest_pathes)

if __name__=='__main__':
    source_dir = './'
    os_walk(source_dir)

