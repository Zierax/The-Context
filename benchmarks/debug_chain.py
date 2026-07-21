#!/usr/bin/env python3
"""Debug: trace the B3→B2→B1→pages chain."""
import sys, os, tempfile, warnings
import numpy as np
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_manager import VirtualMemoryTree
from math_engine import SeededLSH

persist = tempfile.mkdtemp()
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)

# Ingest 5 pages
tokens = []
for i in range(5000):
    tokens.append(f"token_{i}")
pids = list(tree.ingest_stream(iter(tokens)))

print(f"Pages: {len(pids)}")
print(f"B1 beacons: {len(tree.beacon_b1)}")
print(f"B2 beacons: {len(tree.beacon_b2)}")
print(f"B3 beacons: {len(tree.beacon_b3)}")

print(f"\nPending B1: {len(tree._pending_b1_for_b2)}")
print(f"Pending B2: {len(tree._pending_b2_for_b3)}")

# Check B3 → B2 → B1 → pages
print(f"\nb3_to_b2_list: {tree.b3_to_b2_list}")
print(f"b2_to_b1_list: {tree.b2_to_b1_list}")
print(f"b1_to_pages: {tree.b1_to_pages}")

# Check page_to_beacon
print(f"\npage_to_beacon: {tree.page_to_beacon}")

import shutil
shutil.rmtree(persist, ignore_errors=True)
