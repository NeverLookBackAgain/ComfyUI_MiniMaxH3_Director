"""Experimental exact-input singleton image VAE encoding cache.

No text conditioning is shared between different prompts. Enabled only in W6.
"""
import os,hashlib,functools,logging
NODE_CLASS_MAPPINGS={}
if os.environ.get('EXPERIMENT_IMAGE_ENCODE_CACHE')=='1':
 import torch,comfy.sd
 from collections import OrderedDict
 original=comfy.sd.VAE.encode
 @functools.wraps(original)
 def encode_cached(self,pixel_samples,*args,**kwargs):
  if not isinstance(pixel_samples,torch.Tensor) or pixel_samples.ndim!=4 or pixel_samples.shape[0]!=1 or not type(getattr(self,'first_stage_model',None)).__name__.startswith('MiniMaxH3VideoVAE'):
   return original(self,pixel_samples,*args,**kwargs)
  # Pixel values after resize/crop, dtype, shape and call options all identify input.
  if args or kwargs:return original(self,pixel_samples,*args,**kwargs)
  key=(tuple(pixel_samples.shape),str(pixel_samples.dtype),hashlib.sha256(pixel_samples.detach().cpu().contiguous().numpy().tobytes()).digest())
  cache=getattr(self,'_fullopt_image_cache',None)
  if cache is None:cache=OrderedDict();self._fullopt_image_cache=cache
  if key in cache:
   tensor,device=cache[key];cache.move_to_end(key)
   logging.info('Full optimization reference VAE cache HIT shape=%s',tuple(pixel_samples.shape))
   return tensor.to(device=device).clone()
  result=original(self,pixel_samples,*args,**kwargs)
  if isinstance(result,torch.Tensor):
   cache[key]=(result.detach().cpu().clone(),result.device)
   while len(cache)>16:cache.popitem(last=False)
  logging.info('Full optimization reference VAE cache MISS shape=%s',tuple(pixel_samples.shape))
  return result
 comfy.sd.VAE.encode=encode_cached
