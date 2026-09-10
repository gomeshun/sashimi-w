"""Compare installed public WDM counts with the named survivor catalog."""
import json
import numpy as np
from sashimi_w import Subhalos
import argparse
import hashlib
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
options=dict(M0=1e10, redshift=0., dz=.5, zmax=1., N_ma=8, N_herm=2,
             logmamin=5., logmamax=8., N_hermNa=3)
rows=[]
for convention in ['published-q5','standard-t2-q10']:
 for profile in [True,False]:
  model=Subhalos(2.,wdm_power_convention=convention)
  kwargs={**options,'profile_change':profile}
  catalog=model.rs_rhos_catalog_calc(**kwargs)
  expected=float(catalog.weight_final.sum())
  total,x,counts=model.N_sat(**kwargs)
  np.testing.assert_allclose(total,expected,rtol=2e-14)
  ma=catalog.columns['m200_acc']
  np.testing.assert_allclose(counts,[catalog.weight_final[ma>edge].sum() for edge in x],rtol=2e-14,atol=1e-16)
  vtotal,_,_=model.N_sat_Vthres(Vpeak_max=0.,**kwargs)
  np.testing.assert_allclose(vtotal,expected,rtol=2e-14)
  rows.append(dict(convention=convention,profile_change=profile,count=total,velocity_count=vtotal,rows=len(ma)))
result=dict(source_revision=catalog.metadata['sashimi_source_revision'],itamae_source_revision=catalog.metadata['itamae_source_revision'],parameters=options,checks=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+"\n")
print(args.output)
