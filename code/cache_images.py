"""Cache all 9000 images resized to 512x512 uint8 for fast fine-tuning."""
import numpy as np, glob, os, csv, cv2, SimpleITK as sitk
ROOT="datasets/clear_ec_official"; MHA=f"{ROOT}/train_mha"; SZ=512
gt={r["ID"]:r for r in csv.DictReader(open(f"{ROOT}/final_train_ids.csv"))}
files=sorted(glob.glob(f"{MHA}/*.mha"))
ids=[os.path.basename(f).replace('.mha','') for f in files]
files=[f for f,i in zip(files,ids) if i in gt]; ids=[i for i in ids if i in gt]
N=len(files); print("caching",N,"images @",SZ)
arr=np.memmap(f"{ROOT}/img_cache_{SZ}.dat",dtype=np.uint8,mode='w+',shape=(N,SZ,SZ))
for k,f in enumerate(files):
    im=sitk.GetArrayFromImage(sitk.ReadImage(f))  # uint8 (H,W)
    arr[k]=cv2.resize(im,(SZ,SZ),interpolation=cv2.INTER_AREA)
    if k%1500==0: print(k)
arr.flush()
Y=np.array([[float(gt[i]["CD"]),float(gt[i]["CV"]),float(gt[i]["HEX"])] for i in ids],np.float32)
np.savez(f"{ROOT}/cache_meta_{SZ}.npz", Y=Y, ids=np.array(ids), N=N, SZ=SZ)
print("done. cache:",arr.shape)
