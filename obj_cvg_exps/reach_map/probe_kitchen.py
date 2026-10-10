from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom
from simulator.assets.scenes.kitchen import KITCHEN_USD_PATH
st = Usd.Stage.Open(KITCHEN_USD_PATH)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
n = 0
for p in st.Traverse():
    n += 1
    if not p.IsA(UsdGeom.Mesh): continue
    r = cache.ComputeWorldBound(p).ComputeAlignedRange()
    if r.IsEmpty(): continue
    mn, mx = r.GetMin(), r.GetMax()
    # anything overlapping the robot workspace box
    if mx[0] > 8.0 and mn[0] < 10.7 and mx[1] > 5.5 and mn[1] < 7.2 and mx[2] > 0.5 and mn[2] < 1.6:
        print(f"PRIM {p.GetPath()}  x[{mn[0]:.3f},{mx[0]:.3f}] y[{mn[1]:.3f},{mx[1]:.3f}] z[{mn[2]:.3f},{mx[2]:.3f}]", flush=True)
print("PRIMS", n, flush=True)
app.close()
