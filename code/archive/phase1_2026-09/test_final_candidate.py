import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from final_candidate_core import CandidateNet,central512,eval_views,normalize,train_view
from plan_final_candidate import scale_fit


class Tests(unittest.TestCase):
    def test_model_exact_oof_compatibility(self):
        source=ast.parse((Path(__file__).parent/'train_oof_h200.py').read_text())
        node=next(n for n in source.body if getattr(n,'name',None)=='ScalarNet')
        scope=dict(nn=nn,torch=torch,BAG_POOL='mean')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<oof>','exec'),scope)
        class Encoder(nn.Module):
            def forward(self,x): return nn.functional.adaptive_avg_pool2d(x.mean(1,keepdim=True),(2,2)).expand(-1,256,-1,-1)
        original=scope['ScalarNet'](Encoder(),'mean').eval(); candidate=CandidateNet(Encoder()).eval()
        for p in original.parameters(): nn.init.uniform_(p,-.1,.1)
        candidate.load_state_dict(original.state_dict(),strict=True)
        x=torch.rand(3,3,32,32)
        torch.testing.assert_close(original(x),candidate(x),rtol=0,atol=0)

    def test_preprocessing_matches_oof(self):
        import cv2
        source=ast.parse((Path(__file__).parent/'train_oof_h200.py').read_text())
        names={'normalize','eval_origins','dihedral','eval_views'}
        nodes=[n for n in source.body if getattr(n,'name',None) in names]
        pixels=np.random.default_rng(77).integers(0,256,(1,512,512),dtype='uint8')
        scope=dict(np=np,cv2=cv2,clahe=cv2.createCLAHE(2.,(8,8)),CROP=256,SZ=512,CROP_MODE='random',IMGS=pixels)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<views>','exec'),scope)
        np.testing.assert_array_equal(eval_views(pixels[0]),scope['eval_views'](0,1))
        self.assertEqual(train_view(pixels[0]).shape,(3,256,256))
        raw=np.zeros((972,1296),dtype='uint8'); raw[230:742,392:904]=pixels[0]
        np.testing.assert_array_equal(central512(raw),pixels[0])
        with self.assertRaises(ValueError): central512(np.zeros((200,200),dtype='uint8'))

    def test_scale_fit(self):
        p=np.ones((20,3)); y=p*np.array([2.,3.,4.]); y[0,2]=0
        np.testing.assert_array_equal(scale_fit(p,y),[2,3,4])

    def test_file_interface_without_gpu(self):
        import SimpleITK as sitk
        source=Path(__file__).parent.parent/'submission_candidate_20260904/inference.py'
        spec=importlib.util.spec_from_file_location('candidate_interface_test',source)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); inp=root/'input'; image_dir=inp/'images/corneal-specular-microscopy-image'
            image_dir.mkdir(parents=True)
            (inp/'inputs.json').write_text(json.dumps([dict(socket=dict(slug='corneal-specular-microscopy-image'))]))
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((972,1296),dtype='uint8')),str(image_dir/'a.mha'))
            args=['inference.py','--input',str(inp),'--output',str(root/'output'),'--model',str(root/'model')]
            with patch.object(sys,'argv',args),patch.object(module,'predict',return_value=np.array([2700.,.3,.6])):
                module.main()
                self.assertEqual(json.loads((root/'output/cell-density.json').read_text()),2700.)
                self.assertEqual(len(list((root/'output').glob('*.json'))),3)
                with self.assertRaises(FileExistsError): module.main()


if __name__=='__main__': unittest.main()
