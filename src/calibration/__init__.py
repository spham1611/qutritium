# MIT License
#
# Copyright (c) [2023] [son pham, tien nguyen, bach bao]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Import packages and set up IBM backend for calibration"""
from src.backend.backend_ibm import BackEndDict
from src.simple_backend_log import write_log


backend_dict = BackEndDict()
backend, QUBIT_VAL = backend_dict.default_backend()
write_log(backend)
provider = backend_dict.provider

# Constant values coming from the IBM quantum computer. Because those depend
# on computer, we will not save them in the constant.py
ANHAR = backend.qubit_properties(QUBIT_VAL).__getattribute__('anharmonicity')
DEFAULT_F01 = backend.qubit_properties(QUBIT_VAL).frequency
DEFAULT_F12 = DEFAULT_F01 + ANHAR
