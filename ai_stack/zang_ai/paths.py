"""Filesystem locations. Everything else imports paths from here (no hard-coded paths in modules)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai_stack/
WORKSPACE = os.path.dirname(ROOT)                                    # Zang/
CONFIG = os.path.join(ROOT, "config")
OUT = os.path.join(ROOT, "out")
DESCRIPTION = os.path.join(WORKSPACE, "zang_description")
SCENE_XML = os.path.join(DESCRIPTION, "mjcf", "scene.xml")


def config(name):
    return os.path.join(CONFIG, name)
