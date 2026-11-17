"""Flatten the Zang STEP assembly into leaf parts with world-frame meshes and mass properties."""
import sys, pickle, time
import numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopLoc import TopLoc_Location
from OCP.Quantity import Quantity_Color
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

step_path, out_path = sys.argv[1], sys.argv[2]

doc = TDocStd_Document(TCollection_ExtendedString("d"))
reader = STEPCAFControl_Reader()
reader.SetNameMode(True)
reader.SetColorMode(True)
assert reader.ReadFile(step_path) == IFSelect_RetDone
reader.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())


def name(l):
    n = TDataStd_Name()
    return n.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), n) else ""


def color(*labels):
    c = Quantity_Color()
    for l in labels:
        for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
            if type(ct).GetColor_s(l, kind, c):
                return (c.Red(), c.Green(), c.Blue())
    return None


parts = []


def walk(label, path, loc, inherited_color):
    ref = TDF_Label()
    is_ref = st.GetReferredShape_s(label, ref)
    tgt = ref if is_ref else label
    wl = loc * (st.GetLocation_s(label) if is_ref else TopLoc_Location())
    nm = name(label) or name(tgt)
    if nm.startswith("=>"):
        nm = "body"
    p = path + [nm]
    col = color(label, tgt) or inherited_color
    comps = TDF_LabelSequence()
    st.GetComponents_s(tgt, comps)
    if comps.Length():
        for i in range(1, comps.Length() + 1):
            walk(comps.Value(i), p, wl, col)
    else:
        shape = st.GetShape_s(tgt).Moved(wl)
        parts.append(dict(path=p, product=name(tgt), shape=shape, color=col))


roots = TDF_LabelSequence()
st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), [], TopLoc_Location(), None)
print(f"{len(parts)} leaf parts", flush=True)


def mesh(shape):
    BRepMesh_IncrementalMesh(shape, 0.05, False, 0.3, True)
    V, F, off = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            for k in range(1, tri.NbNodes() + 1):
                q = tri.Node(k).Transformed(tr)
                V.append((q.X(), q.Y(), q.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for k in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(k).Get()
                F.append((off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1))
            off += tri.NbNodes()
        exp.Next()
    return np.array(V, float).reshape(-1, 3), np.array(F, int).reshape(-1, 3)


t0 = time.time()
for p in parts:
    s = p["shape"]
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    c = g.CentreOfMass()
    p["volume"] = g.Mass()
    p["centroid"] = np.array([c.X(), c.Y(), c.Z()])
    m = g.MatrixOfInertia()  # about centroid, unit density, mm^5
    p["inertia"] = np.array([[m.Value(r, k) for k in (1, 2, 3)] for r in (1, 2, 3)])
    b = Bnd_Box()
    BRepBndLib.Add_s(s, b)
    p["bbox"] = np.array(b.Get()).reshape(2, 3)
    p["V"], p["F"] = mesh(s)
print(f"meshed in {time.time()-t0:.0f}s", flush=True)

for p in parts:
    del p["shape"]
pickle.dump(dict(parts=parts), open(out_path, "wb"))
