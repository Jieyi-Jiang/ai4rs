from mmengine.fileio import load
res = load('work_dirs/pkl_files/OF_OCDN_np300_boxdn0.5.pkl')
print(type(res), len(res))
print(type(res[0]))
print(res[0])