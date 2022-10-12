#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
This module unfolds the bandstructure of a supercell
to that of a primitive unit cell.

"""

import warnings
import copy
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import ATM_SLOTS, BAS_SLOTS, ATOM_OF, PTR_COORD
from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv3  # noqa
from pyscf import __config__


