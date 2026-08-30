"""Cache central 512x512 NATIVE-resolution crop (cells stay ~27px) for all images."""
import numpy as np, glob, os, csv, SimpleITK as sitk
ROOT="datasets/clear_ec_official"; MHA=f"{ROOT}/train_mha"; SZ=512
gt={r["ID"]:r for r in csv.DictReader(open(f"{ROOT}/final_train_ids.csv"))}
files=sorted(glob.glob(f"{MHA}/*.mha"))
ids=[os.path.basename(f).replace('.mha','') for f in files]
files=[f for f,i in zip(files,ids) if i in gt]; ids=[i for i in ids if i in gt]
N=len(files); print("caching native central",SZ,"for",N)
arr=np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='w+',shape=(N,SZ,SZ))
for k,f in enumerate(files):
    im=sitk.GetArrayFromImage(sitk.ReadImage(f))  # (972,1296) uint8
    H,W=im.shape; r0=(H-SZ)//2; c0=(W-SZ)//2
    arr[k]=im[r0:r0+SZ, c0:c0+SZ]
    if k%1500==0: print(k)
arr.flush()
Y=np.array([[float(gt[i]["CD"]),float(gt[i]["CV"]),float(gt[i]["HEX"])] for i in ids],np.float32)
np.savez(f"{ROOT}/cache_native_meta_{SZ}.npz", Y=Y, ids=np.array(ids), N=N, SZ=SZ)
print("done", arr.shape)
