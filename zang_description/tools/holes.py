"""Collect cylindrical faces (holes / bosses) of selected parts in world coordinates: holes.py STEP OUT.pkl NAME..."""
import sys, pickle
import numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopLoc import TopLoc_Location
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

step_path, out_path, wanted = sys.argv[1], sys.argv[2], set(sys.argv[3:])
doc = TDocStd_Document(TCollection_ExtendedString("d"))
reader = STEPCAFControl_Reader()
reader.SetNameMode(True)
assert reader.ReadFile(step_path) == IFSelect_RetDone
reader.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def name(l):
    n = TDataStd_Name()
    return n.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), n) else ""


result = {}


def walk(label, path, loc):
    ref = TDF_Label()
    is_ref = st.GetReferredShape_s(label, ref)
    tgt = ref if is_ref else label
    wl = loc * (st.GetLocation_s(label) if is_ref else TopLoc_Location())
    p = path + [name(label) or name(tgt)]
    comps = TDF_LabelSequence()
    st.GetComponents_s(tgt, comps)
    if comps.Length():
        for i in range(1, comps.Length() + 1):
            walk(comps.Value(i), p, wl)
        return
    key = "/".join(p[1:])
    if key not in wanted:
        return
    cyl = []
    exp = TopExp_Explorer(st.GetShape_s(tgt).Moved(wl), TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        s = BRepAdaptor_Surface(face)
        if s.GetType() == GeomAbs_Cylinder:
            c = s.Cylinder()
            a = c.Axis()
            b = Bnd_Box()
            BRepBndLib.Add_s(face, b)
            cyl.append(dict(loc=np.array([a.Location().X(), a.Location().Y(), a.Location().Z()]),
                            dir=np.array([a.Direction().X(), a.Direction().Y(), a.Direction().Z()]),
                            r=c.Radius(), bbox=np.array(b.Get()).reshape(2, 3)))
        exp.Next()
    result[key] = cyl


roots = TDF_LabelSequence()
st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), [], TopLoc_Location())
missing = wanted - set(result)
assert not missing, missing
pickle.dump(result, open(out_path, "wb"))
print({k: len(v) for k, v in result.items()})
