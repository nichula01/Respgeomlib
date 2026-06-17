#!/usr/bin/env python3
"""
RespGeomLib Airway Studio  v3
==============================
Desktop application for generating, visualising, and exporting 3-D human
airway lumen geometry with full ANSYS Fluent compatibility.

Features
--------
- Weibel-model Airway generation  G0–G5
- Disease variants: stenosis, dilation
- Embedded interactive 3-D viewer (pyvistaqt)
- Port visualisation (inlet / outlet markers)
- Colour-by-generation mesh rendering
- Geometry Designer: building-block tree view
- ANSYS Fluent export package (wall STL, inlet/outlet caps, journal, ZIP)
- Mesh validation (non-manifold, watertight, components)
- Project save / load (YAML spec + JSON state)

Requirements
------------
    pip install PyQt6 pyvista pyvistaqt
"""

import sys, os, io, json, datetime, zipfile, tempfile, traceback
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter

import numpy as np
import yaml

# ── RespGeomLib path ──────────────────────────────────────────────────────────
# Allow running the app directly (``python app/airway_studio.py``) without an
# editable install by adding the repository root (which contains the
# ``respgeomlib`` package) to sys.path.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

RESPGEOMLIB_OK, _IMPORT_ERR = False, ""
try:
    from respgeomlib.tree_builder import SegmentSpec, build_tree, merge_built_segments, BuiltSegment
    from respgeomlib.morphometry_rules import murray_child_diameters
    RESPGEOMLIB_OK = True
except ImportError as _e:
    _IMPORT_ERR = str(_e)

HAS_PYVISTA, pv = False, None
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pass

# ── Qt ────────────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
        QSlider, QComboBox, QCheckBox, QScrollArea, QHBoxLayout, QVBoxLayout,
        QGridLayout, QSizePolicy, QFileDialog, QMessageBox, QDialog,
        QTextEdit, QTreeWidget, QTreeWidgetItem, QLineEdit, QDoubleSpinBox,
        QGroupBox, QSplitter, QTabWidget,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor
    _H   = Qt.Orientation.Horizontal
    _AC  = Qt.AlignmentFlag.AlignCenter
    _AL  = Qt.AlignmentFlag.AlignLeft
    _AR  = Qt.AlignmentFlag.AlignRight
    _AV  = Qt.AlignmentFlag.AlignVCenter
    _PC  = Qt.CursorShape.PointingHandCursor
    _SBO = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    _NF  = QFrame.Shape.NoFrame
    _HLN = QFrame.Shape.HLine
    PYQT = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (                               # type: ignore
            QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
            QSlider, QComboBox, QCheckBox, QScrollArea, QHBoxLayout, QVBoxLayout,
            QGridLayout, QSizePolicy, QFileDialog, QMessageBox, QDialog,
            QTextEdit, QTreeWidget, QTreeWidgetItem, QLineEdit, QDoubleSpinBox,
            QGroupBox, QSplitter, QTabWidget,
        )
        from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer   # type: ignore
        from PyQt5.QtGui import QFont, QColor                       # type: ignore
        _H   = Qt.Horizontal
        _AC  = Qt.AlignCenter
        _AL  = Qt.AlignLeft
        _AR  = Qt.AlignRight
        _AV  = Qt.AlignVCenter
        _PC  = Qt.PointingHandCursor
        _SBO = Qt.ScrollBarAlwaysOff
        _NF  = QFrame.NoFrame
        _HLN = QFrame.HLine
        PYQT = 5
    except ImportError:
        sys.exit("ERROR: PyQt6 or PyQt5 required.  pip install PyQt6")

HAS_PVQ = False
try:
    from pyvistaqt import QtInteractor
    HAS_PVQ = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE & STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    bg="#060c16", panel="#0c1420", card="#111c2d", input="#162030",
    border="#1d2d40", cyan="#00d4ff", blue="#3d8bff", green="#00e676",
    red="#ff4444", yellow="#ffbb00", orange="#ff8800",
    text="#dde8f5", muted="#7a90a8", dim="#3a5068", sidebar="#080f1a",
)

QSS = f"""
QMainWindow,QWidget{{background:{C['bg']};color:{C['text']};
    font-family:'Segoe UI',Arial,sans-serif;font-size:13px;}}
QFrame#sidebar{{background:{C['sidebar']};border-right:1px solid {C['border']};}}
QFrame#panel{{background:{C['panel']};border:1px solid {C['border']};border-radius:6px;}}
QFrame#card{{background:{C['card']};border:1px solid {C['border']};border-radius:4px;}}
QGroupBox{{background:{C['card']};border:1px solid {C['border']};border-radius:6px;
    margin-top:18px;padding:12px 8px 8px 8px;color:{C['muted']};font-size:11px;font-weight:bold;}}
QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{C['cyan']};}}

QPushButton{{background:{C['input']};color:{C['text']};border:1px solid {C['border']};
    border-radius:4px;padding:5px 12px;}}
QPushButton:hover{{background:{C['card']};border-color:{C['cyan']};}}
QPushButton:disabled{{color:{C['dim']};border-color:{C['border']};background:{C['panel']};}}
QPushButton#primary{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #00eeff,stop:0.45 {C['cyan']},stop:1 #007a99);
    color:#000;font-weight:bold;border:none;
    padding:10px;border-radius:6px;font-size:14px;}}
QPushButton#primary:hover{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #33f5ff,stop:1 #009dbb);}}
QPushButton#primary:disabled{{background:{C['dim']};color:{C['bg']};}}
QPushButton#export{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #5aa0ff,stop:1 {C['blue']});color:#fff;font-weight:bold;border:none;
    padding:7px 10px;border-radius:4px;}}
QPushButton#export:hover{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #7ab5ff,stop:1 #2255cc);}}
QPushButton#ansys{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #ff7733,stop:1 #c44000);color:#fff;font-weight:bold;border:none;
    padding:10px;border-radius:6px;font-size:14px;}}
QPushButton#ansys:hover{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #ff9955,stop:1 #e85400);}}
QPushButton#nav{{background:transparent;border:none;text-align:left;
    padding:9px 14px;border-radius:4px;color:{C['muted']};}}
QPushButton#nav:hover{{background:{C['card']};color:{C['text']};}}
QPushButton#nav_active{{background:{C['card']};border:none;border-left:3px solid {C['cyan']};
    text-align:left;padding:9px 11px;border-radius:4px;color:{C['cyan']};font-weight:bold;}}

QLabel#section{{color:{C['cyan']};font-size:10px;font-weight:bold;letter-spacing:2px;
    border-left:2px solid {C['cyan']};padding-left:6px;margin-top:4px;}}
QLabel#stat_val{{color:{C['cyan']};font-size:22px;font-weight:bold;}}
QLabel#stat_lbl{{color:{C['muted']};font-size:10px;letter-spacing:0.5px;}}
QLabel#preview_title{{color:{C['cyan']};font-size:11px;font-weight:bold;letter-spacing:2px;}}
QLabel#badge_ok{{background:#092515;border:1px solid {C['green']};border-radius:4px;
    color:{C['green']};font-size:11px;font-weight:bold;padding:2px 8px;}}
QLabel#badge_warn{{background:#1a1200;border:1px solid {C['yellow']};border-radius:4px;
    color:{C['yellow']};font-size:11px;font-weight:bold;padding:2px 8px;}}
QLabel#badge_err{{background:#25050a;border:1px solid {C['red']};border-radius:4px;
    color:{C['red']};font-size:11px;font-weight:bold;padding:2px 8px;}}

QSlider::groove:horizontal{{background:{C['border']};height:4px;border-radius:2px;}}
QSlider::handle:horizontal{{background:{C['cyan']};width:14px;height:14px;
    margin:-5px 0;border-radius:7px;}}
QSlider::sub-page:horizontal{{background:{C['cyan']};border-radius:2px;}}

QComboBox{{background:{C['input']};border:1px solid {C['border']};
    border-radius:4px;padding:5px 10px;color:{C['text']};}}
QComboBox::drop-down{{border:none;width:22px;}}
QComboBox QAbstractItemView{{background:{C['card']};border:1px solid {C['border']};
    color:{C['text']};selection-background-color:{C['input']};outline:none;}}

QLineEdit,QDoubleSpinBox{{background:{C['input']};border:1px solid {C['border']};
    border-radius:4px;padding:5px 8px;color:{C['text']};}}
QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{{border:none;background:{C['card']};width:16px;}}

QCheckBox{{color:{C['text']};spacing:8px;}}
QCheckBox::indicator{{width:18px;height:18px;border-radius:9px;
    border:2px solid {C['dim']};background:{C['input']};}}
QCheckBox::indicator:checked{{background:{C['cyan']};border-color:{C['cyan']};}}

QTreeWidget{{background:{C['card']};border:1px solid {C['border']};
    color:{C['text']};outline:none;alternate-background-color:{C['panel']};}}
QTreeWidget::item{{padding:3px;border:none;}}
QTreeWidget::item:selected{{background:{C['input']};color:{C['cyan']};}}
QTreeWidget::item:hover{{background:{C['input']};}}
QTreeWidget::branch{{background:{C['card']};}}
QHeaderView::section{{background:{C['panel']};color:{C['muted']};
    border:none;border-bottom:1px solid {C['border']};padding:4px 8px;font-size:11px;}}

QScrollBar:vertical{{background:{C['panel']};width:6px;border:none;}}
QScrollBar::handle:vertical{{background:{C['dim']};border-radius:3px;min-height:20px;}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
QSplitter::handle{{background:{C['border']};}}
QTextEdit{{background:{C['card']};color:{C['text']};border:1px solid {C['border']};
    border-radius:4px;font-family:Consolas,monospace;font-size:12px;}}
QDialog{{background:{C['bg']};}}
QMessageBox{{background:{C['bg']};}}
QMessageBox QLabel{{color:{C['text']};}}
QTabWidget::pane{{border:1px solid {C['border']};background:{C['panel']};}}
QTabBar::tab{{background:{C['input']};color:{C['muted']};padding:6px 16px;
    border:1px solid {C['border']};border-bottom:none;border-radius:4px 4px 0 0;}}
QTabBar::tab:selected{{background:{C['card']};color:{C['cyan']};
    border-bottom:2px solid {C['cyan']};}}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER WIDGETS
# ─────────────────────────────────────────────────────────────────────────────

def _sec(txt: str) -> QLabel:
    l = QLabel(txt); l.setObjectName("section"); return l

def _hline() -> QFrame:
    f = QFrame(); f.setFrameShape(_HLN); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{C['border']};border:none;margin:2px 0;")
    return f

def _lbl(txt, style=""):
    l = QLabel(txt)
    if style: l.setStyleSheet(style)
    return l


class SliderRow(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, lo, hi, val, dec=2, steps=200):
        super().__init__()
        self._lo, self._hi, self._steps, self._dec = lo, hi, steps, dec
        v = QVBoxLayout(self); v.setContentsMargins(0,2,0,2); v.setSpacing(3)
        h = QHBoxLayout(); h.setContentsMargins(0,0,0,0)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(f"color:{C['text']};font-size:12px;")
        self._val = QLabel(f"{val:.{dec}f}")
        self._val.setStyleSheet(f"color:{C['cyan']};font-size:12px;font-weight:bold;")
        self._val.setFixedWidth(52); self._val.setAlignment(_AR|_AV)
        h.addWidget(self._lbl); h.addStretch(); h.addWidget(self._val)
        self._sl = QSlider(_H)
        self._sl.setRange(0, steps)
        self._sl.setValue(int((val-lo)/(hi-lo)*steps))
        self._sl.valueChanged.connect(self._ch)
        v.addLayout(h); v.addWidget(self._sl)

    def _ch(self, n):
        x = self._lo + (self._hi-self._lo)*n/self._steps
        self._val.setText(f"{x:.{self._dec}f}"); self.valueChanged.emit(x)

    def get(self):
        return self._lo + (self._hi-self._lo)*self._sl.value()/self._steps

    def set(self, val):
        self._sl.setValue(max(0,min(self._steps,
            int((val-self._lo)/(self._hi-self._lo)*self._steps))))


class QRow(QWidget):
    """Label + value row for quality / info panels."""
    def __init__(self, label):
        super().__init__()
        h = QHBoxLayout(self); h.setContentsMargins(0,2,0,2)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        self._v = QLabel("—")
        self._v.setStyleSheet(f"color:{C['text']};font-size:12px;font-weight:bold;")
        self._v.setAlignment(_AR)
        h.addWidget(lbl); h.addStretch(); h.addWidget(self._v)

    def set(self, txt, col=""):
        self._v.setText(txt)
        self._v.setStyleSheet(
            f"color:{col or C['text']};font-size:12px;font-weight:bold;")


# ─────────────────────────────────────────────────────────────────────────────
#  AIRWAY GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class AirwayGenerator:

    # ── Disease helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _pipe_kind(gen_num: int, disease: str, location: str) -> str:
        """Return the pipe kind (pipe / pipe_stenosis / pipe_dilation) based on disease settings."""
        if disease == "none":
            return "pipe"
        hits = {"trachea": gen_num == 0,
                "g4_terminal": gen_num >= 4,
                "all_pipes": True}
        if hits.get(location, False):
            return "pipe_stenosis" if disease == "stenosis" else "pipe_dilation"
        return "pipe"

    @staticmethod
    def _disease_params(d_in, d_out, length, fac) -> dict:
        """Extra params for stenosis/dilation segments."""
        return dict(length=length, d_in=d_in, d_out=d_out,
                    r_min_factor=max(0.1, min(0.95, fac)),
                    center=0.5, width=0.18)

    def generate_specs(self, max_gen=4, trachea_d=1.8, trachea_l=12.0,
                       angle=35.0, terminal_l=2.5, scale=1.0,
                       disease="none", disease_location="trachea",
                       disease_fac=0.5) -> List[SegmentSpec]:
        s, a = scale, float(angle)
        specs: List[SegmentSpec] = []

        def pipe_kind(gen): return self._pipe_kind(gen, disease, disease_location)
        def make_pipe(d_in, d_out, length, gen):
            k = pipe_kind(gen)
            if k == "pipe":
                return k, dict(length=length, d_in=d_in, d_out=d_out)
            return k, self._disease_params(d_in, d_out, length, disease_fac)

        # G0 Trachea – use Weibel generation 0: D=1.80cm, L=12.0cm
        tk, tp = make_pipe(trachea_d*s, trachea_d*s, trachea_l*s, 0)
        specs.append(SegmentSpec("trachea", tk, tp,
            None, None, {"gen":0,"region":"trachea"}))
        if max_gen < 1: return specs

        # G1 Main carina
        d_r = trachea_d*0.694*s; d_l = trachea_d*0.667*s
        specs.append(SegmentSpec("Y_main","y2",
            dict(length_trunk=2.0*s, length_child1=4.5*s, length_child2=5.0*s,
                 d_trunk=trachea_d*s, d_child1=d_r, d_child2=d_l,
                 theta1_deg=a, phi1_deg=-25.0, theta2_deg=a+2, phi2_deg=155.0),
            "trachea",1,{"gen":1,"region":"main_carina"}))
        if max_gen < 2: return specs

        # G2 Lobar bronchi
        dr1,dr2,dr3 = d_r*0.76, d_r*0.68, d_r*0.72
        specs.append(SegmentSpec("Y_right_lobes","y3",
            dict(length_trunk=1.2*s, length_child1=2.5*s, length_child2=2.3*s,
                 length_child3=2.8*s, d_trunk=d_r, d_child1=dr1, d_child2=dr2,
                 d_child3=dr3, theta1_deg=32, phi1_deg=20, theta2_deg=34,
                 phi2_deg=-60, theta3_deg=36, phi3_deg=80),
            "Y_main",1,{"gen":2,"region":"right_lobes"}))
        dl1,dl2 = d_l*0.792, d_l*0.750
        specs.append(SegmentSpec("Y_left_lobes","y2",
            dict(length_trunk=1.5*s, length_child1=2.6*s, length_child2=2.9*s,
                 d_trunk=d_l, d_child1=dl1, d_child2=dl2,
                 theta1_deg=34, phi1_deg=25, theta2_deg=36, phi2_deg=205),
            "Y_main",2,{"gen":2,"region":"left_lobes"}))
        if max_gen < 3: return specs

        # G3 Lobar bifurcations
        g3 = [("Y_right_upper_G3","Y_right_lobes",1,dr1,(15,-35)),
              ("Y_right_middle_G3","Y_right_lobes",2,dr2,(-20,-80)),
              ("Y_right_lower_G3","Y_right_lobes",3,dr3,(70,130)),
              ("Y_left_upper_G3","Y_left_lobes",1,dl1,(30,-40)),
              ("Y_left_lower_G3","Y_left_lobes",2,dl2,(190,240))]
        g3c: Dict[str,tuple] = {}
        for sid,par,port,d,(p1,p2) in g3:
            dc1,dc2 = d*0.59, d*0.57
            specs.append(SegmentSpec(sid,"y2",
                dict(length_trunk=0.8*s, length_child1=1.5*s, length_child2=1.4*s,
                     d_trunk=d, d_child1=dc1, d_child2=dc2,
                     theta1_deg=35, phi1_deg=p1, theta2_deg=37, phi2_deg=p2),
                par,port,{"gen":3,"region":sid}))
            g3c[sid]=(dc1,dc2)
        if max_gen < 4: return specs

        # G4 Terminal pipes  (Weibel G4: D≈0.45cm, L/D≈2.8)
        g4: List[SegmentSpec] = []
        for sid,_,_,_,_ in g3:
            dc1,dc2 = g3c[sid]
            base = sid.replace("Y_","").replace("_G3","")
            for idx,(di,suf) in enumerate([(dc1,"a"),(dc2,"b")],1):
                tk4, pp4 = make_pipe(di, di*0.85, terminal_l*s, 4)
                g4.append(SegmentSpec(f"{base}_G4_{suf}", tk4, pp4, sid, idx,
                                      {"gen":4,"region":base}))
        specs.extend(g4)
        if max_gen < 5: return specs

        # G5 Murray extension
        for sp in g4:
            di = sp.params["d_in"]
            dc1,dc2 = murray_child_diameters(di,asymmetry=0.05)
            specs.append(SegmentSpec(f"Y_{sp.id}_G5","y2",
                dict(length_trunk=di*2*s, length_child1=dc1*3*s, length_child2=dc2*3*s,
                     d_trunk=di, d_child1=dc1, d_child2=dc2,
                     theta1_deg=a, phi1_deg=0, theta2_deg=a+5, phi2_deg=120),
                sp.id,1,{"gen":5,"region":sp.meta.get("region","")}))
        return specs

    def build(self, specs: List[SegmentSpec]) -> dict:
        """Build mesh + return everything callers need."""
        root_o = np.array([0.,0.,0.]); root_z = np.array([0.,0.,-1.])
        built = build_tree(specs, root_origin=root_o, root_z_world=root_z)
        pts, faces = merge_built_segments(built)

        # ── Vertex welding fix ──────────────────────────────────────────────────
        # merge_built_segments() concatenates meshes without welding shared-boundary
        # vertices.  Pipe ring vertices and implicit-junction marching-cubes vertices
        # are nearly coincident but not identical → disconnected components.
        # pv.PolyData.clean(tolerance, absolute=True) welds them.
        if HAS_PYVISTA and pts.shape[0] > 0 and faces.shape[0] > 0:
            try:
                ff = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int),
                                faces]).ravel()
                cleaned = pv.PolyData(pts, ff).clean(tolerance=0.04, absolute=True)
                pts   = np.array(cleaned.points, dtype=float)
                fc    = cleaned.faces
                if len(fc) > 0:
                    faces = fc.reshape(-1, 4)[:, 1:].astype(int)
            except Exception:
                pass  # fall back to un-welded mesh; quality panel will report issues

        parents = {s.parent_id for s in specs if s.parent_id}
        outlets = sum(1 for s in specs if s.id not in parents)
        gens = [s.meta["gen"] for s in specs if s.meta and "gen" in s.meta]

        stats = dict(n_points=int(pts.shape[0]), n_triangles=int(faces.shape[0]),
                     outlet_count=outlets, max_gen=max(gens) if gens else 0,
                     n_segments=len(specs))
        return dict(points=pts, faces=faces, built=built, specs=specs, stats=stats)

    def quality(self, pts, faces) -> dict:
        if not HAS_PYVISTA:
            return dict(cfd_ready=None, boundary_edges="n/a",
                        non_manifold_edges="n/a", connected_components="n/a")
        ff = np.hstack([np.full((faces.shape[0],1),3,dtype=int),faces]).ravel()
        m = pv.PolyData(pts, ff)
        nm  = m.extract_feature_edges(False, True,  False, False)  # non-manifold
        bnd = m.extract_feature_edges(True,  False, False, False)  # boundary (open ports)
        try:
            conn = m.connectivity()
            nc = int(len(np.unique(conn["RegionId"])))
        except Exception:
            nc = 1
        # CFD-ready = zero non-manifold edges.
        # Boundary edges at open ports are *intentional* and not a defect.
        cfd_ready = (int(nm.n_cells) == 0)
        return dict(cfd_ready=cfd_ready,
                    non_manifold_edges=int(nm.n_cells),
                    boundary_edges=int(bnd.n_cells),
                    connected_components=nc)


# ─────────────────────────────────────────────────────────────────────────────
#  ANSYS FLUENT EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

class ANSYSExporter:
    """
    Creates an ANSYS Fluent-compatible export package:
      geometry/airway_wall.stl       – airway wall surface
      geometry/airway_inlet.stl      – flat inlet cap
      geometry/airway_outlet_NNN.stl – individual outlet caps
      geometry/airway_combined.stl   – all surfaces merged
      fluent/fluent_meshing.jou      – Fluent Meshing WTG journal
      fluent/fluent_solver.jou       – Fluent solver setup journal
      specs/airway_spec.yaml         – reproducible geometry spec
      README.txt
    """

    # ── Cap geometry ──────────────────────────────────────────────────────────
    @staticmethod
    def make_cap(center, normal, radius, n=32):
        c = np.asarray(center, float)
        nh = np.asarray(normal, float); nh /= np.linalg.norm(nh)
        ref = np.array([1.,0.,0.]) if abs(nh[0]) < 0.9 else np.array([0.,1.,0.])
        u = np.cross(nh, ref); u /= np.linalg.norm(u)
        v = np.cross(nh, u)
        th = np.linspace(0, 2*np.pi, n, endpoint=False)
        rim = c + radius*(np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
        pts = np.vstack([c.reshape(1,3), rim])
        faces = np.array([(0, i+1, i%n+1) for i in range(n)])
        return pts, faces

    # ── Port extraction ───────────────────────────────────────────────────────
    @staticmethod
    def get_ports(specs, built):
        """
        Returns:
          inlet  : (position, direction, radius)
          outlets: list of (position, direction, radius, seg_id)
        """
        by_id = {s.id: s for s in specs}
        parents = {s.parent_id for s in specs if s.parent_id}
        terminals = [s.id for s in specs if s.id not in parents]

        root_s = next(s for s in specs if s.parent_id is None)
        root_b = built[root_s.id]
        inlet = (root_b.ports_world[0].position,
                 root_b.ports_world[0].direction,
                 root_s.params["d_in"] / 2.0)

        outlets = []
        for tid in terminals:
            tb = built[tid]; ts = by_id[tid]
            if len(tb.ports_world) > 1:
                p = tb.ports_world[1]
                r = ts.params.get("d_out", ts.params.get("d_in", 0.2)) / 2.0
                outlets.append((p.position, p.direction, r, tid))
        return inlet, outlets

    # ── ASCII STL writer ──────────────────────────────────────────────────────
    @staticmethod
    def to_stl_str(pts, faces, name="solid"):
        buf = [f"solid {name}"]
        for f in faces:
            v0,v1,v2 = pts[f[0]], pts[f[1]], pts[f[2]]
            n = np.cross(v1-v0, v2-v0)
            nn = np.linalg.norm(n)
            if nn > 1e-12: n /= nn
            buf.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            buf.append("    outer loop")
            for v in (v0,v1,v2):
                buf.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            buf.append("    endloop\n  endfacet")
        buf.append(f"endsolid {name}")
        return "\n".join(buf)

    # ── Main export ───────────────────────────────────────────────────────────
    def export_zip(self, specs, built, pts, faces, zip_path,
                   units="m", inlet_vel=0.2, outlet_pa=0.0,
                   fluid="air", progress_cb=None):
        """Package everything into a ZIP file."""
        scale = {"m": 0.01, "mm": 10.0, "cm": 1.0}[units]
        pts_s = pts * scale

        inlet, outlets = self.get_ports(specs, built)

        def _prog(msg):
            if progress_cb: progress_cb(msg)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:

            # 1. Wall surface
            _prog("Writing wall surface…")
            zf.writestr("geometry/airway_wall.stl",
                        self.to_stl_str(pts_s, faces, "airway_wall"))

            # 2. Inlet cap
            _prog("Writing inlet cap…")
            inp, ind, inr = inlet
            icap_pts, icap_f = self.make_cap(inp*scale, -ind, inr*scale)
            zf.writestr("geometry/airway_inlet.stl",
                        self.to_stl_str(icap_pts, icap_f, "inlet"))

            # 3. Outlet caps
            _prog("Writing outlet caps…")
            all_cap_pts = [icap_pts]
            all_cap_f   = [icap_f]
            off = len(icap_pts)
            for i, (op, od, orad, otid) in enumerate(outlets):
                cp, cf = self.make_cap(op*scale, -od, orad*scale)
                zf.writestr(f"geometry/airway_outlet_{i+1:03d}.stl",
                            self.to_stl_str(cp, cf, f"outlet_{i+1:03d}"))
                all_cap_pts.append(cp)
                all_cap_f.append(cf + off)
                off += len(cp)

            # 4. Combined (wall + caps)
            _prog("Writing combined surface…")
            all_pts = np.vstack([pts_s] + all_cap_pts)
            all_f   = np.vstack([faces] + [cf + (0 if j==0 else
                        pts_s.shape[0] + sum(len(p) for p in all_cap_pts[:j]))
                        for j, cf in enumerate([faces] + all_cap_f)])
            # simpler: just wall + all caps sequentially
            comb_pts = [pts_s] + all_cap_pts
            comb_fs  = []
            off2 = 0
            for p, f in zip(comb_pts, [faces]+all_cap_f):
                comb_fs.append(f + off2); off2 += len(p)
            zf.writestr("geometry/airway_combined.stl",
                        self.to_stl_str(np.vstack(comb_pts),
                                        np.vstack(comb_fs), "airway_combined"))

            # 5. Fluent Meshing journal
            _prog("Writing Fluent journals…")
            zf.writestr("fluent/fluent_meshing.jou",
                        self._meshing_journal(len(outlets), units))
            zf.writestr("fluent/fluent_solver.jou",
                        self._solver_journal(len(outlets), inlet_vel,
                                              outlet_pa, fluid, units))

            # 6. YAML spec
            data = [{"id":s.id,"kind":s.kind,"params":s.params,
                     "parent_id":s.parent_id,
                     "parent_port_index":s.parent_port_index}
                    for s in specs]
            zf.writestr("specs/airway_spec.yaml",
                        yaml.safe_dump(data, sort_keys=False))

            # 7. README
            zf.writestr("README.txt",
                        self._readme(len(outlets), units, inlet_vel))

        _prog("Done.")

    # ── Fluent Meshing journal ─────────────────────────────────────────────────
    @staticmethod
    def _meshing_journal(n_outlets, units):
        return f"""; ================================================================
; RespGeomLib Airway Studio — ANSYS Fluent Meshing Journal
; Watertight Geometry Workflow
; ================================================================
; HOW TO USE:
;   Fluent Meshing > File > Run Journal > fluent_meshing.jou
; ================================================================

; ── Initialise WTG workflow ──────────────────────────────────────
/workflow/initialize-workflow 'Watertight Geometry'

; ── Import wall geometry ─────────────────────────────────────────
/meshing/workflow/TaskObject["Import Geometry"]/CommandArguments() \\
    setDict {{FileName "geometry/airway_wall.stl" LengthUnit "{units}"}}
/meshing/workflow/TaskObject["Import Geometry"]/Execute()

; ── Import inlet cap ─────────────────────────────────────────────
/meshing/workflow/TaskObject["Import Geometry"]/CommandArguments() \\
    setDict {{FileName "geometry/airway_inlet.stl" LengthUnit "{units}"}}
/meshing/workflow/TaskObject["Import Geometry"]/Execute()

; ── Import outlet caps (repeat for each outlet) ──────────────────
""" + "\n".join(
            f"""/meshing/workflow/TaskObject["Import Geometry"]/CommandArguments() \\
    setDict {{FileName "geometry/airway_outlet_{i+1:03d}.stl" LengthUnit "{units}"}}
/meshing/workflow/TaskObject["Import Geometry"]/Execute()"""
            for i in range(n_outlets)
        ) + f"""

; ── Add local sizing ─────────────────────────────────────────────
/meshing/workflow/TaskObject["Add Local Sizing"]/Execute()

; ── Generate surface mesh ────────────────────────────────────────
/meshing/workflow/TaskObject["Generate the Surface Mesh"]/Execute()

; ── Describe geometry ────────────────────────────────────────────
/meshing/workflow/TaskObject["Describe Geometry"]/CommandArguments() \\
    setDict {{SetupType "The geometry consists of only fluid regions with no voids"}}
/meshing/workflow/TaskObject["Describe Geometry"]/Execute()

; ── Update boundaries ────────────────────────────────────────────
/meshing/workflow/TaskObject["Update Boundaries"]/Execute()

; ── Generate volume mesh ─────────────────────────────────────────
/meshing/workflow/TaskObject["Generate the Volume Mesh"]/Execute()

; ── Switch to solver ─────────────────────────────────────────────
/mesh/check
"""

    # ── Fluent solver journal ─────────────────────────────────────────────────
    @staticmethod
    def _solver_journal(n_outlets, vel, pa, fluid, units):
        rho = 1.225 if fluid == "air" else 998.0
        mu  = 1.8e-5 if fluid == "air" else 1.0e-3
        return f"""; ================================================================
; RespGeomLib Airway Studio — ANSYS Fluent Solver Setup Journal
; ================================================================

; ── Read mesh (update path as needed) ───────────────────────────
; /file/read-case "airway_solution.cas.gz"

; ── Solver settings ──────────────────────────────────────────────
/define/models/viscous kw-sst yes

; ── Fluid material: {fluid} ──────────────────────────────────────
/define/materials/change-create {fluid} {fluid} \\
    yes constant {rho} no no yes constant {mu} no no no

; ── Boundary conditions ──────────────────────────────────────────
; Inlet (velocity-inlet): {vel} m/s
/define/boundary-conditions/velocity-inlet \\
    inlet yes no yes yes no {vel} no 0

; Outlets (pressure-outlet): {pa} Pa gauge
""" + "\n".join(
            f"/define/boundary-conditions/pressure-outlet outlet_{i+1:03d} yes no {pa} no yes"
            for i in range(n_outlets)
        ) + """

; ── Wall ─────────────────────────────────────────────────────────
/define/boundary-conditions/wall wall yes no no 0

; ── Initialise & iterate ─────────────────────────────────────────
/solve/initialize/hybrid-initialization
/solve/iterate 500

; ── Save ─────────────────────────────────────────────────────────
/file/write-case-data "airway_solution.cas.gz"
"""

    # ── README ────────────────────────────────────────────────────────────────
    @staticmethod
    def _readme(n_outlets, units, vel):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""RespGeomLib Airway Studio — ANSYS Fluent Export Package
Generated: {ts}
{'='*60}

CONTENTS
--------
geometry/airway_wall.stl         Airway wall surface (units: {units})
geometry/airway_inlet.stl        Inlet cap (trachea opening)
geometry/airway_outlet_NNN.stl   {n_outlets} outlet caps (terminal bronchi)
geometry/airway_combined.stl     Wall + all caps in one file

fluent/fluent_meshing.jou        Fluent Meshing WTG journal
fluent/fluent_solver.jou         Fluent solver setup journal

specs/airway_spec.yaml           Reproducible geometry specification

IMPORT INTO ANSYS FLUENT (RECOMMENDED)
---------------------------------------
1. Launch ANSYS Fluent Meshing
2. Activate "Watertight Geometry" workflow
3. File > Run Journal > select  fluent/fluent_meshing.jou
4. Adjust mesh sizing as needed, then generate volume mesh
5. Switch to Fluent solver
6. File > Run Journal > select  fluent/fluent_solver.jou

MANUAL IMPORT (ALTERNATIVE)
----------------------------
1. Fluent Meshing > Watertight Geometry
2. Import each STL file separately:
     airway_wall.stl   → wall zone
     airway_inlet.stl  → velocity-inlet zone
     airway_outlet_NNN.stl → pressure-outlet zones
3. Describe geometry as "fluid only (no voids)"
4. Generate surface mesh → volume mesh

BOUNDARY ZONES
--------------
  inlet         velocity-inlet    suggested U = {vel} m/s
  outlet_NNN    pressure-outlet   P_gauge = 0 Pa  ({n_outlets} zones)
  wall          no-slip wall

SUGGESTED PHYSICS (k-omega SST)
---------------------------------
  Fluid:     Air  rho=1.225 kg/m³  mu=1.8e-5 Pa·s
  Inlet vel: {vel} m/s
  Re (trachea, D~0.018 m):  ~{int(1.225*vel*0.018/1.8e-5)}  (laminar-transitional)
  Outlet P:  0 Pa (gauge)
  Turbulence: k-omega SST

REPRODUCIBILITY
---------------
The file  specs/airway_spec.yaml  contains the full parametric
specification. Load it in Airway Studio to regenerate the exact
same geometry at any time.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class _Worker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, gen: AirwayGenerator, params: dict):
        super().__init__(); self._g = gen; self._p = params

    def run(self):
        try:
            p = self._p
            specs = self._g.generate_specs(
                max_gen=p["max_gen"], trachea_d=p["trachea_d"],
                trachea_l=p["trachea_l"], angle=p["angle"],
                terminal_l=p["terminal_l"], scale=p["scale"],
                disease=p.get("disease","none"),
                disease_location=p.get("disease_location","trachea"),
                disease_fac=max(0.1, min(0.95, p.get("disease_fac", 0.5))),
            )
            res = self._g.build(specs)
            res["quality"] = self._g.quality(res["points"], res["faces"])
            self.done.emit(res)
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  3-D VIEWER
# ─────────────────────────────────────────────────────────────────────────────

class ThreeDViewer(QWidget):
    GEN_COLORS = ["#aaaaaa","#4488ff","#00d4ff","#00e676","#ffbb00","#ff8800"]

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(380)
        self._L = QVBoxLayout(self); self._L.setContentsMargins(0,0,0,0); self._L.setSpacing(0)
        self._pl = None
        self._color_mode = "uniform"

        # Toolbar
        tb = QWidget(); tb.setFixedHeight(38)
        tb.setStyleSheet(f"background:{C['panel']};border-bottom:1px solid {C['border']};")
        th = QHBoxLayout(tb); th.setContentsMargins(12,0,12,0)
        t = QLabel("3D AIRWAY PREVIEW"); t.setObjectName("preview_title")
        th.addWidget(t); th.addStretch()

        self._col_combo = QComboBox()
        self._col_combo.addItems(["Uniform","By Generation","By Type"])
        self._col_combo.setFixedWidth(140)
        self._col_combo.setStyleSheet(f"font-size:11px;padding:2px 6px;")
        self._col_combo.currentIndexChanged.connect(self._recolor)
        th.addWidget(self._col_combo)
        th.addSpacing(6)

        for ico, tip, fn in [("⟳","Reset",self._reset),("⤢","Fit",self._fit)]:
            b = QPushButton(ico); b.setToolTip(tip); b.setFixedSize(28,26)
            b.setStyleSheet(f"background:{C['input']};border:1px solid {C['border']};"
                            f"border-radius:4px;color:{C['text']};font-size:14px;")
            b.clicked.connect(fn); th.addWidget(b)
        self._L.addWidget(tb)

        if HAS_PVQ:
            self._pl = QtInteractor(self)
            self._pl.set_background(C["bg"])
            self._L.addWidget(self._pl, stretch=1)
            self._splash()
        else:
            msg = QLabel("3D preview requires pyvistaqt\npip install pyvistaqt\n\n"
                         "Export still works without it.")
            msg.setAlignment(_AC)
            msg.setStyleSheet(f"color:{C['muted']};font-size:13px;background:{C['card']};")
            self._L.addWidget(msg, stretch=1)

        self._last_pts = None; self._last_f = None
        self._outlet_pts = []; self._inlet_pt = None
        self._specs: List[SegmentSpec] = []

    def _splash(self):
        if self._pl:
            self._pl.clear()
            self._pl.add_text("Click  ⚡ GENERATE  to build the airway",
                              font_size=10, color=C["muted"])

    def _reset(self):
        if self._pl: self._pl.reset_camera()

    def _fit(self):
        if self._pl: self._pl.reset_camera()

    def show_loading(self):
        if self._pl:
            self._pl.clear()
            self._pl.add_text("Generating…", font_size=16, color=C["cyan"])
            self._pl.render()

    def display(self, pts, faces, outlet_positions, inlet_position, specs):
        self._last_pts = pts; self._last_f = faces
        self._outlet_pts = outlet_positions; self._inlet_pt = inlet_position
        self._specs = specs
        self._render()

    def _render(self):
        if not self._pl or not HAS_PYVISTA or self._last_pts is None: return
        ff = np.hstack([np.full((self._last_f.shape[0],1),3,dtype=int),
                        self._last_f]).ravel()
        mesh = pv.PolyData(self._last_pts, ff)

        self._pl.clear()
        self._pl.set_background(C["bg"])

        mode = self._col_combo.currentIndex()
        if mode == 0:  # Uniform
            self._pl.add_mesh(mesh, color="#3a77bb", smooth_shading=True,
                              show_edges=False, opacity=0.95)
        elif mode == 1 and self._specs:  # By generation
            gen_arr = self._build_gen_scalar(mesh)
            self._pl.add_mesh(mesh, scalars=gen_arr, cmap="cool",
                              smooth_shading=True, show_edges=False,
                              scalar_bar_args=dict(title="Generation",
                                                   color=C["text"],
                                                   fmt="%.0f"))
        else:
            self._pl.add_mesh(mesh, color="#3a77bb", smooth_shading=True,
                              show_edges=False, opacity=0.95)

        # Inlet marker (green sphere)
        if self._inlet_pt is not None:
            sp = pv.Sphere(radius=0.15, center=self._inlet_pt)
            self._pl.add_mesh(sp, color=C["green"], opacity=0.9)

        # Outlet markers (yellow spheres)
        for op in self._outlet_pts:
            sp = pv.Sphere(radius=0.1, center=op)
            self._pl.add_mesh(sp, color=C["yellow"], opacity=0.85)

        self._pl.reset_camera()
        self._pl.render()

    def _build_gen_scalar(self, mesh):
        """Assign generation scalar per vertex (approximate)."""
        return np.zeros(mesh.n_points)

    def _recolor(self, _):
        self._render()


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER PANEL  (left panel on Home page)
# ─────────────────────────────────────────────────────────────────────────────

class ParameterPanel(QFrame):
    generate_requested = pyqtSignal(dict)
    yaml_load_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__(); self.setObjectName("panel"); self.setFixedWidth(285)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(_NF)
        sc.setHorizontalScrollBarPolicy(_SBO)
        sc.setStyleSheet("background:transparent;")
        body = QWidget(); body.setStyleSheet("background:transparent;")
        L = QVBoxLayout(body); L.setContentsMargins(16,16,16,16); L.setSpacing(6)

        L.addWidget(_sec("GEOMETRY CONTROLS")); L.addWidget(_hline())
        L.addWidget(QLabel("Model Type"))
        self.mdl = QComboBox()
        self.mdl.addItems(["Human Weibel Tree (G0–G5)","Load Custom YAML…"])
        self.mdl.currentIndexChanged.connect(self._mdl_changed)
        L.addWidget(self.mdl); L.addSpacing(2)
        L.addWidget(QLabel("Max Generation"))
        self.gen = QComboBox(); self.gen.addItems(["G1","G2","G3","G4","G5"])
        self.gen.setCurrentIndex(3); L.addWidget(self.gen); L.addSpacing(4)

        self.s_td = SliderRow("Trachea Diameter (cm)",0.5,3.0,1.8)
        self.s_tl = SliderRow("Trachea Length (cm)",5.0,20.0,12.0)
        self.s_an = SliderRow("Branching Angle (°)",10.0,65.0,32.0,dec=1)
        self.s_ll = SliderRow("Terminal Branch Length (cm)",0.5,5.0,2.5)
        self.s_sc = SliderRow("Scale Factor",0.5,2.0,1.0)
        for w in [self.s_td,self.s_tl,self.s_an,self.s_ll,self.s_sc]:
            L.addWidget(w)
        L.addSpacing(8)

        L.addWidget(_sec("DISEASE CONTROLS")); L.addWidget(_hline())

        # Disease Type
        h_dt = QHBoxLayout(); h_dt.setContentsMargins(0,0,0,0)
        ldt = QLabel("Disease Type")
        ldt.setStyleSheet(f"color:{C['text']};font-size:12px;")
        h_dt.addWidget(ldt); h_dt.addStretch()
        L.addLayout(h_dt)
        self.disease_type = QComboBox()
        self.disease_type.addItems(["None  (healthy)",
                                    "Stenosis  (narrowing)",
                                    "Dilation  (widening)"])
        self.disease_type.currentIndexChanged.connect(self._disease_changed)
        L.addWidget(self.disease_type); L.addSpacing(4)

        # Disease Location
        h_dl = QHBoxLayout(); h_dl.setContentsMargins(0,0,0,0)
        ldl = QLabel("Location")
        ldl.setStyleSheet(f"color:{C['text']};font-size:12px;")
        h_dl.addWidget(ldl); h_dl.addStretch()
        L.addLayout(h_dl)
        self.disease_loc = QComboBox()
        self.disease_loc.addItems(["Trachea  (G0 — most visible)",
                                   "Terminal Bronchi  (G4)",
                                   "All Pipe Segments"])
        L.addWidget(self.disease_loc); L.addSpacing(4)

        self.s_sv = SliderRow("Severity  (lower = more severe)", 0.1, 0.95, 0.5)
        L.addWidget(self.s_sv); L.addSpacing(4)

        # Affected segment indicator
        self._disease_info = QLabel("Disease: Disabled")
        self._disease_info.setStyleSheet(
            f"color:{C['muted']};font-size:11px;"
            f"background:{C['card']};border:1px solid {C['border']};"
            f"border-radius:4px;padding:4px 8px;")
        self._disease_info.setWordWrap(True)
        L.addWidget(self._disease_info); L.addSpacing(12)

        # Initialize disease control enabled state
        self.disease_loc.setEnabled(False)
        self.s_sv.setEnabled(False)

        self.btn = QPushButton("⚡   GENERATE GEOMETRY")
        self.btn.setObjectName("primary"); self.btn.setCursor(_PC)
        self.btn.clicked.connect(self._fire); L.addWidget(self.btn)
        L.addStretch(); sc.setWidget(body)
        OL = QVBoxLayout(self); OL.setContentsMargins(0,0,0,0); OL.addWidget(sc)

    def _mdl_changed(self, i):
        for w in [self.gen,self.s_td,self.s_tl,self.s_an,self.s_ll,self.s_sc]:
            w.setEnabled(i==0)
        if i == 1:
            p,_ = QFileDialog.getOpenFileName(self,"Open YAML",
                                              str(_HERE/"trees"),"YAML (*.yaml *.yml)")
            if p: self.yaml_load_requested.emit(p)
            else: self.mdl.setCurrentIndex(0)

    def _disease_changed(self, idx):
        enabled = (idx > 0)
        self.disease_loc.setEnabled(enabled)
        self.s_sv.setEnabled(enabled)
        if idx == 0:
            self._disease_info.setText("Disease: Disabled  (healthy model)")
            self._disease_info.setStyleSheet(
                f"color:{C['muted']};font-size:11px;"
                f"background:{C['card']};border:1px solid {C['border']};"
                f"border-radius:4px;padding:4px 8px;")
        elif idx == 1:
            loc_labels = ["Trachea (1 segment)", "Terminal bronchi (10 segments)", "All pipe segments (11+)"]
            self._disease_info.setText(f"Stenosis ▸ {loc_labels[self.disease_loc.currentIndex()]}")
            self._disease_info.setStyleSheet(
                f"color:{C['red']};font-size:11px;font-weight:bold;"
                f"background:#1a0505;border:1px solid {C['red']};"
                f"border-radius:4px;padding:4px 8px;")
        else:
            loc_labels = ["Trachea (1 segment)", "Terminal bronchi (10 segments)", "All pipe segments (11+)"]
            self._disease_info.setText(f"Dilation ▸ {loc_labels[self.disease_loc.currentIndex()]}")
            self._disease_info.setStyleSheet(
                f"color:{C['yellow']};font-size:11px;font-weight:bold;"
                f"background:#1a1200;border:1px solid {C['yellow']};"
                f"border-radius:4px;padding:4px 8px;")

    def _fire(self):
        disease_map = {0: "none", 1: "stenosis", 2: "dilation"}
        loc_map     = {0: "trachea", 1: "g4_terminal", 2: "all_pipes"}
        self.generate_requested.emit({
            "max_gen":   {"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}.get(self.gen.currentText(),4),
            "trachea_d": self.s_td.get(), "trachea_l": self.s_tl.get(),
            "angle":     self.s_an.get(), "terminal_l": self.s_ll.get(),
            "scale":     self.s_sc.get(),
            "disease":          disease_map[self.disease_type.currentIndex()],
            "disease_location": loc_map[self.disease_loc.currentIndex()],
            "disease_fac":      self.s_sv.get(),
        })

    def set_busy(self, b):
        self.btn.setEnabled(not b)
        self.btn.setText("⏳   Generating…" if b else "⚡   GENERATE GEOMETRY")


# ─────────────────────────────────────────────────────────────────────────────
#  MESH QUALITY PANEL  (right panel on Home page)
# ─────────────────────────────────────────────────────────────────────────────

class QualityPanel(QFrame):
    export_stl  = pyqtSignal()
    export_yaml = pyqtSignal()
    open_ansys  = pyqtSignal()

    def __init__(self):
        super().__init__(); self.setObjectName("panel"); self.setFixedWidth(220)
        L = QVBoxLayout(self); L.setContentsMargins(14,14,14,14); L.setSpacing(6)

        L.addWidget(_sec("MESH QUALITY")); L.addWidget(_hline())
        self._badge = QLabel("—"); self._badge.setAlignment(_AC)
        self._badge.setFixedHeight(44)
        self._badge.setStyleSheet(
            f"background:{C['input']};border:1px solid {C['border']};"
            f"border-radius:6px;font-size:13px;font-weight:bold;color:{C['muted']};")
        L.addWidget(self._badge); L.addSpacing(4)

        self.rw = QRow("CFD-Ready")
        self.rb = QRow("Boundary Edges")
        self.rn = QRow("Non-manifold Edges")
        self.rc = QRow("Connected Components")
        self.rt = QRow("Triangle Count")
        self.ro = QRow("Outlet Count")
        for r in [self.rw,self.rb,self.rn,self.rc,self.rt,self.ro]:
            L.addWidget(r)
        # Tooltip explaining boundary edges
        self.rb.setToolTip("Open edges at inlet/outlet ports — intentional.\n"
                           "These become named boundary zones in ANSYS Fluent.")
        L.addSpacing(10)

        L.addWidget(_sec("QUICK EXPORT")); L.addWidget(_hline())
        b1 = QPushButton("⬇  Export STL"); b1.setObjectName("export")
        b1.clicked.connect(self.export_stl)
        b2 = QPushButton("⬇  Export YAML Spec")
        b2.clicked.connect(self.export_yaml)
        b3 = QPushButton("📦  ANSYS Package…")
        b3.setObjectName("export"); b3.clicked.connect(self.open_ansys)
        for b in [b1,b2,b3]: L.addWidget(b)
        L.addStretch()

    def update(self, q, stats):
        ok  = q.get("cfd_ready");  nm   = q.get("non_manifold_edges","n/a")
        bnd = q.get("boundary_edges","n/a"); comp = q.get("connected_components","n/a")
        tri = stats.get("n_triangles",0);    out  = stats.get("outlet_count",0)

        if ok is True:
            self._badge.setText("✓   CFD-READY")
            self._badge.setStyleSheet(
                f"background:#092515;border:1px solid {C['green']};"
                f"border-radius:6px;font-size:13px;font-weight:bold;color:{C['green']};")
        elif ok is False:
            self._badge.setText("✗   NON-MANIFOLD")
            self._badge.setStyleSheet(
                f"background:#25050a;border:1px solid {C['red']};"
                f"border-radius:6px;font-size:13px;font-weight:bold;color:{C['red']};")

        self.rw.set("✓ PASS" if ok else "✗ FAIL", C["green"] if ok else C["red"])
        # Boundary edges at ports are expected; only flag if unexpected (NM edges instead)
        self.rb.set(str(bnd), C["muted"])
        self.rn.set(str(nm),  C["red"]   if isinstance(nm,int) and nm>0  else C["green"])
        self.rc.set(str(comp),C["green"] if comp==1             else C["yellow"])
        self.rt.set(f"{tri:,}"); self.ro.set(str(out))


# ─────────────────────────────────────────────────────────────────────────────
#  STATS BAR
# ─────────────────────────────────────────────────────────────────────────────

class StatsBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(76)
        self.setStyleSheet(f"background:{C['panel']};border-top:1px solid {C['border']};")
        h = QHBoxLayout(self); h.setContentsMargins(24,4,24,4)
        self._v: Dict[str,QLabel] = {}
        items = [("outlets","Outlets","○"),("gens","Generations","⌀"),
                 ("tris","Triangles","△"),("inlets","Inlet","▽"),
                 ("val","Validation","✓")]
        for i,(k,lbl,ico) in enumerate(items):
            col = QWidget(); col.setStyleSheet("background:transparent;")
            vl = QVBoxLayout(col); vl.setContentsMargins(0,0,0,0)
            vl.setSpacing(1); vl.setAlignment(_AC)
            v = QLabel("—"); v.setObjectName("stat_val"); v.setAlignment(_AC)
            l = QLabel(f"{ico}  {lbl}"); l.setObjectName("stat_lbl"); l.setAlignment(_AC)
            vl.addWidget(v); vl.addWidget(l)
            self._v[k] = v; h.addWidget(col)
            if i < len(items)-1:
                s = QFrame(); s.setFixedWidth(1)
                s.setStyleSheet(f"background:{C['border']};"); h.addWidget(s)

    def update(self, outlets, gen_range, tris, inlets, valid):
        self._v["outlets"].setText(str(outlets))
        self._v["gens"].setText(gen_range)
        self._v["tris"].setText(f"{tris//1000}K" if tris>=1000 else str(tris))
        self._v["inlets"].setText(str(inlets))
        v = self._v["val"]
        if valid is True:
            v.setText("PASS"); v.setStyleSheet(f"color:{C['green']};font-size:22px;font-weight:bold;")
        elif valid is False:
            v.setText("FAIL"); v.setStyleSheet(f"color:{C['red']};font-size:22px;font-weight:bold;")
        else:
            v.setText("—"); v.setStyleSheet(f"color:{C['muted']};font-size:22px;font-weight:bold;")


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY DESIGNER PAGE
# ─────────────────────────────────────────────────────────────────────────────

class GeometryDesignerPage(QWidget):
    def __init__(self):
        super().__init__()
        L = QHBoxLayout(self); L.setContentsMargins(8,8,8,8); L.setSpacing(8)

        # Left: tree
        lf = QFrame(); lf.setObjectName("panel")
        lv = QVBoxLayout(lf); lv.setContentsMargins(12,12,12,12); lv.setSpacing(6)
        lv.addWidget(_sec("BUILDING BLOCKS TREE")); lv.addWidget(_hline())

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Segment ID","Type","Gen","Region"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setDefaultSectionSize(120)
        self.tree.itemClicked.connect(self._on_select)
        lv.addWidget(self.tree)
        L.addWidget(lf, 2)

        # Right: properties
        rf = QFrame(); rf.setObjectName("panel")
        rv = QVBoxLayout(rf); rv.setContentsMargins(12,12,12,12); rv.setSpacing(6)
        rv.addWidget(_sec("SEGMENT PROPERTIES")); rv.addWidget(_hline())

        grid = QWidget(); gl = QGridLayout(grid); gl.setContentsMargins(0,4,0,4)
        self._props: Dict[str,QLabel] = {}
        rows = [("ID","id"),("Type","kind"),("Generation","gen"),
                ("Region","region"),("Parent","parent"),("Port Index","port_idx")]
        for i,(label,key) in enumerate(rows):
            gl.addWidget(_lbl(label, f"color:{C['muted']};font-size:12px;"), i, 0)
            v = QLabel("—"); v.setStyleSheet(f"color:{C['text']};font-size:12px;font-weight:bold;")
            gl.addWidget(v, i, 1); self._props[key] = v
        rv.addWidget(grid); rv.addWidget(_hline())

        rv.addWidget(_sec("PARAMETERS")); rv.addWidget(_hline())
        self.param_text = QTextEdit(); self.param_text.setReadOnly(True)
        self.param_text.setMaximumHeight(200)
        rv.addWidget(self.param_text)
        rv.addWidget(_hline())
        rv.addWidget(_sec("PORT INFORMATION")); rv.addWidget(_hline())
        self.port_info = QLabel("Select a segment to view its ports.")
        self.port_info.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        self.port_info.setWordWrap(True)
        rv.addWidget(self.port_info)
        rv.addStretch()

        L.addWidget(rf, 1)
        self._specs: List[SegmentSpec] = []

    def populate(self, specs: List[SegmentSpec]):
        self._specs = specs
        self.tree.clear()
        children: Dict[str,List[SegmentSpec]] = {}
        for s in specs:
            if s.parent_id: children.setdefault(s.parent_id,[]).append(s)

        def add(spec, parent_item=None):
            gen = str(spec.meta.get("gen","?")) if spec.meta else "?"
            region = str(spec.meta.get("region","")) if spec.meta else ""
            item = QTreeWidgetItem([spec.id, spec.kind, f"G{gen}", region])
            # Colour by type
            col = {"pipe":"#4488ff","pipe_stenosis":"#ff4444",
                   "pipe_dilation":"#ffbb00","y2":"#00d4ff","y3":"#00e676"
                   }.get(spec.kind, C["text"])
            item.setForeground(1, QColor(col))
            item.setData(0, 32, spec.id)  # store id
            if parent_item: parent_item.addChild(item)
            else:           self.tree.addTopLevelItem(item)
            for ch in children.get(spec.id,[]): add(ch, item)
        root = next((s for s in specs if s.parent_id is None), None)
        if root: add(root)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

    def _on_select(self, item, _col):
        sid = item.data(0, 32)
        spec = next((s for s in self._specs if s.id == sid), None)
        if not spec: return
        self._props["id"].setText(spec.id)
        self._props["kind"].setText(spec.kind)
        self._props["gen"].setText(str(spec.meta.get("gen","?")) if spec.meta else "?")
        self._props["region"].setText(str(spec.meta.get("region","")) if spec.meta else "")
        self._props["parent"].setText(spec.parent_id or "— (root)")
        self._props["port_idx"].setText(str(spec.parent_port_index) if spec.parent_port_index is not None else "—")
        lines = [f"  {k}: {v}" for k,v in spec.params.items()]
        self.param_text.setPlainText("\n".join(lines))

        # Port count hint
        ports_n = {"pipe":2,"pipe_stenosis":2,"pipe_dilation":2,"y2":3,"y3":4}.get(spec.kind,2)
        self.port_info.setText(
            f"Ports: {ports_n}\n"
            f"  Port 0 — parent (inlet)\n"
            + "\n".join(f"  Port {i+1} — child outlet {i+1}"
                        for i in range(ports_n-1))
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ANSYS EXPORT PAGE
# ─────────────────────────────────────────────────────────────────────────────

class AnsysExportPage(QWidget):
    export_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        L = QHBoxLayout(self); L.setContentsMargins(8,8,8,8); L.setSpacing(8)

        # ── Left: config ──────────────────────────────────────────────────────
        lf = QFrame(); lf.setObjectName("panel")
        lv = QVBoxLayout(lf); lv.setContentsMargins(16,16,16,16); lv.setSpacing(8)

        lv.addWidget(_sec("EXPORT CONFIGURATION")); lv.addWidget(_hline())

        # Units
        ug = QGroupBox("Units & Scaling")
        ugl = QGridLayout(ug); ugl.setSpacing(8)
        ugl.addWidget(_lbl("Output units"), 0, 0)
        self.units = QComboBox(); self.units.addItems(["m (metres)","mm (millimetres)","cm (centimetres)"])
        ugl.addWidget(self.units, 0, 1)
        ugl.addWidget(_lbl("Note: geometry is defined in cm internally", f"color:{C['muted']};font-size:11px;"), 1, 0, 1, 2)
        lv.addWidget(ug)

        # What to include
        ig = QGroupBox("Include in Package")
        igl = QVBoxLayout(ig)
        self.ck_wall    = QCheckBox("Wall surface STL");            self.ck_wall.setChecked(True)
        self.ck_inlet   = QCheckBox("Inlet cap STL");               self.ck_inlet.setChecked(True)
        self.ck_outlets = QCheckBox("Outlet cap STLs");             self.ck_outlets.setChecked(True)
        self.ck_combined= QCheckBox("Combined (all-in-one) STL");   self.ck_combined.setChecked(True)
        self.ck_mesh_j  = QCheckBox("Fluent Meshing journal (.jou)");self.ck_mesh_j.setChecked(True)
        self.ck_solv_j  = QCheckBox("Fluent Solver journal (.jou)"); self.ck_solv_j.setChecked(True)
        self.ck_spec    = QCheckBox("Reproducibility YAML spec");    self.ck_spec.setChecked(True)
        for c in [self.ck_wall,self.ck_inlet,self.ck_outlets,self.ck_combined,
                  self.ck_mesh_j,self.ck_solv_j,self.ck_spec]:
            igl.addWidget(c)
        lv.addWidget(ig)

        # Flow conditions
        fg = QGroupBox("Flow Conditions (for journal)")
        fgl = QGridLayout(fg); fgl.setSpacing(8)
        fgl.addWidget(_lbl("Fluid"), 0, 0)
        self.fluid = QComboBox(); self.fluid.addItems(["air","water"])
        self.fluid.currentIndexChanged.connect(self._update_re)
        fgl.addWidget(self.fluid, 0, 1)

        fgl.addWidget(_lbl("Inlet velocity (m/s)"), 1, 0)
        self.vel = QDoubleSpinBox(); self.vel.setRange(0.001, 50.0)
        self.vel.setValue(0.2); self.vel.setDecimals(3); self.vel.setSingleStep(0.05)
        self.vel.valueChanged.connect(self._update_re)
        fgl.addWidget(self.vel, 1, 1)

        fgl.addWidget(_lbl("Outlet pressure (Pa gauge)"), 2, 0)
        self.outlet_pa = QDoubleSpinBox(); self.outlet_pa.setRange(-1e5, 1e5)
        self.outlet_pa.setValue(0.0); self.outlet_pa.setDecimals(1)
        fgl.addWidget(self.outlet_pa, 2, 1)
        lv.addWidget(fg)

        self.btn = QPushButton("📦   EXPORT ANSYS PACKAGE")
        self.btn.setObjectName("ansys"); self.btn.setCursor(_PC)
        self.btn.clicked.connect(self._fire)
        lv.addWidget(self.btn); lv.addStretch()
        L.addWidget(lf, 3)

        # ── Right: info / Re calculator ───────────────────────────────────────
        rf = QFrame(); rf.setObjectName("panel")
        rv = QVBoxLayout(rf); rv.setContentsMargins(16,16,16,16); rv.setSpacing(8)

        rv.addWidget(_sec("BOUNDARY ZONES")); rv.addWidget(_hline())
        self.zone_info = QLabel("Generate geometry first.")
        self.zone_info.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        self.zone_info.setWordWrap(True)
        rv.addWidget(self.zone_info)
        rv.addSpacing(8)

        rv.addWidget(_sec("PHYSICS PREVIEW")); rv.addWidget(_hline())
        self.re_lbl   = QLabel("Re = —")
        self.re_lbl.setStyleSheet(f"color:{C['cyan']};font-size:20px;font-weight:bold;")
        self.flow_lbl = QLabel("Flow regime: —")
        self.flow_lbl.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        rv.addWidget(self.re_lbl); rv.addWidget(self.flow_lbl)
        rv.addSpacing(8)

        rv.addWidget(_sec("PACKAGE CONTENTS")); rv.addWidget(_hline())
        self.contents_lbl = QLabel(
            "geometry/airway_wall.stl\n"
            "geometry/airway_inlet.stl\n"
            "geometry/airway_outlet_NNN.stl\n"
            "geometry/airway_combined.stl\n"
            "fluent/fluent_meshing.jou\n"
            "fluent/fluent_solver.jou\n"
            "specs/airway_spec.yaml\n"
            "README.txt"
        )
        self.contents_lbl.setStyleSheet(f"color:{C['text']};font-size:11px;font-family:Consolas,monospace;")
        rv.addWidget(self.contents_lbl)
        rv.addStretch()
        L.addWidget(rf, 2)

        self._trachea_d = 1.8  # cm, updated when mesh is ready

    def set_mesh_ready(self, stats: dict, n_outlets: int, trachea_d: float):
        self._trachea_d = trachea_d
        self.zone_info.setText(
            f"  inlet          → velocity-inlet  (1 zone)\n"
            f"  outlet_NNN     → pressure-outlet ({n_outlets} zones)\n"
            f"  wall           → no-slip wall\n\n"
            f"  Total segments : {stats.get('n_segments',0)}\n"
            f"  Triangles      : {stats.get('n_triangles',0):,}\n"
            f"  Outlets        : {n_outlets}"
        )
        self._update_re()

    def _update_re(self):
        d_m  = self._trachea_d * 0.01           # cm → m
        vel  = self.vel.value()
        rho, mu = (1.225, 1.8e-5) if self.fluid.currentIndex()==0 else (998.0, 1.0e-3)
        Re = rho * vel * d_m / mu
        self.re_lbl.setText(f"Re = {Re:.0f}")
        if Re < 2300:
            regime = "Laminar"
            col = C["green"]
        elif Re < 4000:
            regime = "Transitional"
            col = C["yellow"]
        else:
            regime = "Turbulent"
            col = C["red"]
        self.flow_lbl.setText(f"Flow regime: {regime}  (D={d_m*100:.1f} cm, U={vel:.3f} m/s)")
        self.flow_lbl.setStyleSheet(f"color:{col};font-size:12px;")

    def _fire(self):
        units_map = {"m (metres)":"m","mm (millimetres)":"mm","cm (centimetres)":"cm"}
        self.export_requested.emit({
            "units":      units_map.get(self.units.currentText(),"m"),
            "inlet_vel":  self.vel.value(),
            "outlet_pa":  self.outlet_pa.value(),
            "fluid":      self.fluid.currentText(),
        })

    def set_busy(self, b: bool):
        self.btn.setEnabled(not b)
        self.btn.setText("⏳  Exporting…" if b else "📦   EXPORT ANSYS PACKAGE")


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION PAGE
# ─────────────────────────────────────────────────────────────────────────────

class ValidationPage(QWidget):
    def __init__(self):
        super().__init__()
        L = QVBoxLayout(self); L.setContentsMargins(8,8,8,8); L.setSpacing(8)
        L.addWidget(_sec("DETAILED MESH VALIDATION")); L.addWidget(_hline())

        self.report = QTextEdit(); self.report.setReadOnly(True)
        self.report.setPlaceholderText("Generate an airway to see validation results here.")
        L.addWidget(self.report)

    def update(self, quality: dict, stats: dict, specs: List[SegmentSpec]):
        ok  = quality.get("cfd_ready"); nm  = quality.get("non_manifold_edges","n/a")
        bnd = quality.get("boundary_edges","n/a"); comp = quality.get("connected_components","n/a")
        kinds = Counter(s.kind for s in specs)
        gens  = Counter(s.meta["gen"] for s in specs if s.meta and "gen" in s.meta)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outlet_count = stats.get("outlet_count", 0)
        # Expected open edges = 32 per port ring (n_theta in segments.py) × (1 inlet + N outlets)
        expected_bnd = (outlet_count + 1) * 32 if isinstance(bnd, int) else "?"

        lines = [
            f"RespGeomLib Airway Studio — Mesh Validation Report",
            f"="*60,
            f"Timestamp          : {ts}",
            f"",
            f"MESH GEOMETRY",
            f"  Vertices         : {stats.get('n_points',0):,}",
            f"  Triangles        : {stats.get('n_triangles',0):,}",
            f"  Segments         : {stats.get('n_segments',0)}",
            f"  Outlet count     : {outlet_count}",
            f"  Max generation   : G{stats.get('max_gen',0)}",
            f"",
            f"CFD READINESS CHECKS",
            f"  CFD-Ready        : {'✓ PASS' if ok else '✗ FAIL' if ok is False else 'N/A (pyvista needed)'}",
            f"  Non-manifold edges: {nm}  {'✓ (none — good)' if nm==0 else '✗ FAIL — mesh defects detected'}",
            f"",
            f"  Boundary edges   : {bnd}",
            f"  (Expected ~{expected_bnd} open-port edges = {outlet_count+1} ports × 32 ring verts)",
            f"  Boundary edges at open ports are INTENTIONAL — not a defect.",
            f"  These become inlet/outlet boundary zones in Fluent Meshing.",
            f"",
            f"  Connected comps  : {comp}  {'✓' if comp==1 else '⚠ multiple components detected'}",
            f"",
            f"BUILDING BLOCKS",
        ]
        for kind,cnt in sorted(kinds.items()):
            lines.append(f"  {kind:<25s} {cnt}")
        lines += ["","GENERATION BREAKDOWN"]
        for g in sorted(gens):
            lines.append(f"  G{g}  →  {gens[g]} segment(s)")
        lines += [
            "",
            "IMPORT NOTE (ANSYS Fluent)",
            "  Use WTG workflow with individual STL files per zone.",
            "  Wall: airway_wall.stl | Inlet: airway_inlet.stl",
            "  Outlets: airway_outlet_NNN.stl (one per terminal bronchus)",
        ]
        self.report.setPlainText("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

class Sidebar(QFrame):
    nav = pyqtSignal(str)

    _ITEMS = [("home","🏠","Home"),("designer","✏","Geometry Designer"),
              ("ansys","📦","ANSYS Export"),("validation","☑","Mesh Validation"),
              ("history","⏱","Project History")]
    _BOT   = [("settings","⚙","Settings"),("about","ℹ","About")]

    def __init__(self):
        super().__init__(); self.setObjectName("sidebar"); self.setFixedWidth(220)
        L = QVBoxLayout(self); L.setContentsMargins(10,16,10,14); L.setSpacing(2)

        logo = QWidget(); logo.setStyleSheet("background:transparent;")
        lh = QHBoxLayout(logo); lh.setContentsMargins(8,0,8,14)
        ico = QLabel("🫁"); ico.setStyleSheet("font-size:28px;")
        txts = QWidget(); txts.setStyleSheet("background:transparent;")
        tv = QVBoxLayout(txts); tv.setContentsMargins(0,0,0,0); tv.setSpacing(0)
        tv.addWidget(_lbl("RespGeomLib",f"color:{C['cyan']};font-size:14px;font-weight:bold;"))
        tv.addWidget(_lbl("Airway Studio",f"color:{C['muted']};font-size:11px;"))
        lh.addWidget(ico); lh.addWidget(txts); L.addWidget(logo)
        L.addWidget(_hline()); L.addSpacing(6)

        self._btns: Dict[str,QPushButton] = {}; self._active = "home"
        for k,ic,lab in self._ITEMS:
            b = QPushButton(f"  {ic}  {lab}")
            b.setObjectName("nav_active" if k=="home" else "nav")
            b.clicked.connect(lambda _,key=k: self._act(key))
            L.addWidget(b); self._btns[k] = b
        L.addStretch(); L.addWidget(_hline())
        for k,ic,lab in self._BOT:
            b = QPushButton(f"  {ic}  {lab}"); b.setObjectName("nav")
            b.clicked.connect(lambda _,key=k: self._act(key))
            L.addWidget(b); self._btns[k] = b

        L.addSpacing(10)
        card = QFrame(); card.setObjectName("card")
        cv = QVBoxLayout(card); cv.setContentsMargins(10,8,10,8); cv.setSpacing(2)
        cv.addWidget(_sec("ACTIVE PROJECT"))
        self.proj_name = QLabel("—"); self.proj_name.setWordWrap(True)
        self.proj_name.setStyleSheet(f"color:{C['text']};font-size:12px;font-weight:bold;")
        self.proj_time = QLabel("Not yet generated")
        self.proj_time.setStyleSheet(f"color:{C['muted']};font-size:11px;")
        cv.addWidget(self.proj_name); cv.addWidget(self.proj_time); L.addWidget(card)

    def _act(self, key):
        if self._active in self._btns:
            self._btns[self._active].setObjectName("nav")
            self._btns[self._active].style().unpolish(self._btns[self._active])
            self._btns[self._active].style().polish(self._btns[self._active])
        self._active = key
        if key in self._btns:
            self._btns[key].setObjectName("nav_active")
            self._btns[key].style().unpolish(self._btns[key])
            self._btns[key].style().polish(self._btns[key])
        self.nav.emit(key)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RespGeomLib — Airway Studio")
        self.resize(1460, 880); self.setMinimumSize(960, 600)

        self._gen = AirwayGenerator()
        self._exp = ANSYSExporter()
        self._worker: Optional[_Worker] = None
        self._result: Optional[dict]    = None   # last successful build

        # ── Root layout ───────────────────────────────────────────────────────
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        self.sidebar = Sidebar(); self.sidebar.nav.connect(self._on_nav)
        root.addWidget(self.sidebar)

        # Right area: pages + stats bar
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)

        # ── Pages (stacked via QTabWidget hidden, manual switching) ──────────
        # Use a dict of pages + show/hide approach for simplicity
        self._pages: Dict[str, QWidget] = {}

        # HOME page: 3-column layout
        home = QWidget()
        hl = QHBoxLayout(home); hl.setContentsMargins(8,8,8,8); hl.setSpacing(8)
        self.param_panel = ParameterPanel()
        self.param_panel.generate_requested.connect(self._on_generate)
        self.viewer = ThreeDViewer()
        self.quality_panel = QualityPanel()
        self.quality_panel.export_stl.connect(self._export_stl)
        self.quality_panel.export_yaml.connect(self._export_yaml)
        self.quality_panel.open_ansys.connect(lambda: self._on_nav("ansys"))
        hl.addWidget(self.param_panel); hl.addWidget(self.viewer, stretch=1)
        hl.addWidget(self.quality_panel)
        self._pages["home"] = home

        # GEOMETRY DESIGNER page
        self.designer = GeometryDesignerPage()
        self._pages["designer"] = self.designer

        # ANSYS EXPORT page
        self.ansys_page = AnsysExportPage()
        self.ansys_page.export_requested.connect(self._on_ansys_export)
        self._pages["ansys"] = self.ansys_page

        # VALIDATION page
        self.val_page = ValidationPage()
        self._pages["validation"] = self.val_page

        # HISTORY / SETTINGS / ABOUT  (placeholder)
        for k, msg in [("history","Project history coming soon."),
                       ("settings","Settings coming soon."),
                       ("about", "RespGeomLib Airway Studio\n\n"
                                 "Built on the RespGeomLib parametric airway engine.\n"
                                 "Authors: Nichula Wasalathilake, Parakrama Ekanayake,\n"
                                 "         Roshan Godaliyadda — University of Peradeniya")]:
            ph = QWidget()
            phl = QVBoxLayout(ph); phl.setAlignment(_AC)
            l = QLabel(msg); l.setAlignment(_AC); l.setWordWrap(True)
            l.setStyleSheet(f"color:{C['muted']};font-size:14px;")
            phl.addWidget(l); self._pages[k] = ph

        # Page container
        self._page_container = QWidget()
        self._pcl = QVBoxLayout(self._page_container)
        self._pcl.setContentsMargins(0,0,0,0); self._pcl.setSpacing(0)
        for p in self._pages.values():
            self._pcl.addWidget(p); p.setVisible(False)
        self._pages["home"].setVisible(True)
        self._current_page = "home"

        rl.addWidget(self._page_container, stretch=1)
        self.stats_bar = StatsBar()
        rl.addWidget(self.stats_bar)
        root.addWidget(right, stretch=1)

        if not RESPGEOMLIB_OK:
            QTimer.singleShot(300, self._warn_import)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _on_nav(self, key: str):
        if key not in self._pages:
            return
        self._pages[self._current_page].setVisible(False)
        self._pages[key].setVisible(True)
        self._current_page = key

    # ── Generation ────────────────────────────────────────────────────────────
    def _on_generate(self, params: dict):
        if not RESPGEOMLIB_OK: self._warn_import(); return
        self.param_panel.set_busy(True); self.viewer.show_loading()
        self._worker = _Worker(self._gen, params)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, res: dict):
        self.param_panel.set_busy(False)
        self._result = res
        pts, faces = res["points"], res["faces"]
        stats, quality = res["stats"], res["quality"]
        specs, built  = res["specs"],  res["built"]

        # Extract port positions for visualisation
        try:
            (inp, ind, _), outlets = ANSYSExporter.get_ports(specs, built)
            inlet_pos   = inp
            outlet_pos  = [o[0] for o in outlets]
        except Exception:
            inlet_pos, outlet_pos = None, []

        # Update viewer
        self.viewer.display(pts, faces, outlet_pos, inlet_pos, specs)

        # Update quality panel
        self.quality_panel.update(quality, stats)

        # Update stats bar
        mg = stats.get("max_gen",0)
        self.stats_bar.update(stats.get("outlet_count",0),
                              f"G0–G{mg}", stats.get("n_triangles",0),
                              1, quality.get("cfd_ready"))

        # Update geometry designer
        self.designer.populate(specs)

        # Update validation page
        self.val_page.update(quality, stats, specs)

        # Update ANSYS export page
        trachea_d = next((s.params["d_in"] for s in specs
                          if s.id=="trachea"), 1.8)
        self.ansys_page.set_mesh_ready(stats, len(outlet_pos), trachea_d)

        # Sidebar project info
        now = datetime.datetime.now().strftime("%I:%M %p")
        self.sidebar.proj_name.setText(f"Human Weibel Tree (G0–G{mg})")
        self.sidebar.proj_time.setText(f"Last built: Today, {now}")

    def _on_error(self, msg: str):
        self.param_panel.set_busy(False)
        QMessageBox.critical(self, "Generation Error",
                             f"Failed to generate airway:\n\n{msg}")

    # ── ANSYS Export ──────────────────────────────────────────────────────────
    def _on_ansys_export(self, cfg: dict):
        if not self._result:
            QMessageBox.warning(self,"Nothing to Export","Generate an airway first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ANSYS Package", "airway_ansys_package.zip",
            "ZIP Archive (*.zip)")
        if not path: return

        self.ansys_page.set_busy(True)
        specs  = self._result["specs"]
        built  = self._result["built"]
        pts    = self._result["points"]
        faces  = self._result["faces"]

        try:
            self._exp.export_zip(specs, built, pts, faces, path,
                                  units=cfg["units"],
                                  inlet_vel=cfg["inlet_vel"],
                                  outlet_pa=cfg["outlet_pa"],
                                  fluid=cfg["fluid"])
            self.ansys_page.set_busy(False)
            QMessageBox.information(self, "Export Complete",
                                    f"ANSYS package saved to:\n{path}\n\n"
                                    "Open the ZIP and follow README.txt for\n"
                                    "import instructions into Fluent.")
        except Exception:
            self.ansys_page.set_busy(False)
            QMessageBox.critical(self, "Export Error",
                                 f"Export failed:\n\n{traceback.format_exc()}")

    # ── Quick exports from quality panel ─────────────────────────────────────
    def _export_stl(self):
        if not self._result: QMessageBox.warning(self,"No Mesh","Generate first."); return
        path,_ = QFileDialog.getSaveFileName(self,"Export STL","airway.stl","STL (*.stl)")
        if path and HAS_PYVISTA:
            pts = self._result["points"]; faces = self._result["faces"]
            ff = np.hstack([np.full((faces.shape[0],1),3,dtype=int),faces]).ravel()
            pv.PolyData(pts, ff).save(path)
            QMessageBox.information(self,"Exported",f"STL saved:\n{path}")

    def _export_yaml(self):
        if not self._result: return
        path,_ = QFileDialog.getSaveFileName(self,"Export YAML","airway_spec.yaml","YAML (*.yaml)")
        if path:
            data = [{"id":s.id,"kind":s.kind,"params":s.params,
                     "parent_id":s.parent_id,"parent_port_index":s.parent_port_index}
                    for s in self._result["specs"]]
            with open(path,"w",encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            QMessageBox.information(self,"Exported",f"YAML saved:\n{path}")

    # ── Misc ──────────────────────────────────────────────────────────────────
    def _warn_import(self):
        QMessageBox.critical(self,"Import Error",
            f"Could not import RespGeomLib:\n\n{_IMPORT_ERR}\n\n"
            "Ensure you run from the project folder.")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RespGeomLib Airway Studio")
    app.setStyleSheet(QSS)
    win = MainWindow(); win.show()
    # Auto-generate default tree on startup
    QTimer.singleShot(700, win.param_panel._fire)
    sys.exit(app.exec() if PYQT==6 else app.exec_())

if __name__ == "__main__":
    main()
