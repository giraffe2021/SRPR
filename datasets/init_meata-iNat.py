import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

sys.path.append('..')

data_path = "/data/giraffe/0_FSL/data/inat2017"
origin_path = "/data/giraffe/0_FSL/data"
# origin_path = os.path.join(data_path, 'train_val_images')
imgfolder = 'inat2017_224x224'
img_path = os.path.join(data_path, imgfolder)
rel_path = os.path.join('..', '..', '..', imgfolder)
inat_path = os.path.join(data_path, 'meta_iNat')
tier_path = os.path.join(data_path, 'tiered_meta_iNat')

os.makedirs(img_path, exist_ok=True)
os.makedirs(inat_path, exist_ok=True)
os.makedirs(tier_path, exist_ok=True)

with open(os.path.join(data_path, 'train_2017_bboxes.json')) as f:
    allinfo = json.load(f)
annolist = allinfo['annotations']

annodict = dict()  # im_id to list of box_ids
boxdict = dict()  # box_id to box coords
catdict = dict()  # dict of numerical category codes / labels to corresponding list of image ids
for d in annolist:
    im = d['image_id']
    boxid = d['id']
    cat = d['category_id']

    # Add box_id to image entry
    if im in annodict:
        annodict[im].append(boxid)
    else:
        annodict[im] = [boxid]

    # Add mapping from box_id to box
    boxdict[boxid] = d['bbox']

    # Add image to category set
    if cat in catdict:
        catdict[cat].add(im)
    else:
        catdict[cat] = set([im])

# assemble im_id -> filepath dictionary

namelist = allinfo['images']
keys = []
vals = []
for d in namelist:
    keys.append(d['id'])
    vals.append(os.path.join(origin_path, d['file_name']))
pather = dict(zip(keys, vals))

# Pare down the category dictionary to the desired size

clist = list(catdict.keys())
for c in clist:
    if len(catdict[c]) < 50 or len(catdict[c]) > 1000:
        catdict.pop(c)

supercat = dict()
for d in allinfo['categories']:
    catid = d['id']
    if catid in catdict:
        sc = d['supercategory']
        if sc in supercat:
            supercat[sc].append(catid)
        else:
            supercat[sc] = [catid, ]

# shrink images
catlist = list(catdict.keys())
boxdict_shrunk = dict()  # abbreviated path -> [box corners]
pather_shrunk = dict()  # im_id -> new path (relative, for symlinks)
print('Shrinking images to 224x224 ...')


def process_image(c, imkey, catdict, pather, img_path, rel_path, pather_shrunk):
    catpath = os.path.join(img_path, str(c))
    os.makedirs(catpath, exist_ok=True)
    path = pather[imkey]
    fname = os.path.join(path[path.rfind(os.path.sep) + 1:path.rfind('.')] + '.jpg')
    newpath = os.path.join(catpath, fname)
    pather_shrunk[imkey] = os.path.join(rel_path, str(c), fname)
    # image = cv2.imread(path)
    # image = cv2.resize(image, (224, 224))
    # cv2.imwrite(newpath, image)


with ThreadPoolExecutor(max_workers=36) as executor:
    # 提交任务给线程池
    futures = []
    for c in tqdm(catlist):
        for imkey in catdict[c]:
            future = executor.submit(process_image, c, imkey, catdict, pather, img_path, rel_path, pather_shrunk)
            futures.append(future)
    # 等待所有任务完成
    for future in tqdm(futures, desc="Processing images"):
        future.result()


def makedataset(traincatlist, testcatlist, datapath, catdict, pather):
    def makesplit(catlist, datapath, split, catdict, pather, imsplit):
        splitpath = os.path.join(datapath, split)
        os.makedirs(splitpath, exist_ok=True)
        for c in catlist:
            # For each category:
            catpath = os.path.join(splitpath, str(c))
            if not os.path.exists(catpath):
                os.makedirs(catpath)
            ims = list(catdict[c])
            ims = imsplit(ims)
            for imkey in ims:
                path = pather[imkey]
                newpath = os.path.join(catpath, path[path.rfind(os.path.sep) + 1:path.rfind('.')] + '.jpg')
                os.symlink(path, newpath)

    os.makedirs(os.path.join(datapath, 'val'), exist_ok=True)
    makesplit(traincatlist, datapath, 'train', catdict, pather, lambda x: x)
    makesplit(testcatlist, datapath, 'test', catdict, pather, lambda x: x)
    makesplit(testcatlist, datapath, 'refr', catdict, pather, lambda x: x[:len(x) // 5])
    makesplit(testcatlist, datapath, 'query', catdict, pather, lambda x: x[len(x) // 5:])


# meta-iNat
print('Organizing meta-iNat ...')
split_folder = os.path.abspath('./')
traincatlist = torch.load(os.path.join(split_folder, 'meta_iNat_split/meta_iNat_traincats.pth'))
testcatlist = torch.load(os.path.join(split_folder, 'meta_iNat_split/meta_iNat_testcats.pth'))
makedataset(traincatlist, testcatlist, inat_path, catdict, pather_shrunk)

# tiered meta-iNat
print('Organizing tiered meta-iNat ...')
traincatlist = (supercat['Animalia'] + supercat['Aves'] + supercat['Reptilia'] + supercat['Amphibia']
                + supercat['Mammalia'] + supercat['Actinopterygii'] + supercat['Mollusca'])
testcatlist = supercat['Insecta'] + supercat['Arachnida']
makedataset(traincatlist, testcatlist, tier_path, catdict, pather_shrunk)

print('Organizing complete!')
