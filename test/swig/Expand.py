# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License") you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
#
# This file is part of DNN compiler maintained at
# https://github.com/ai-techsystems/dnnCompiler


import common

import deepC.dnnc as dc
import numpy as np
import unittest

class ExpandTest(unittest.TestCase):
    def setUp(self):
        self.len = 3
        self.np_a = np.random.rand(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))
    def test_Expand1D (self):
        npr = np.broadcast_to(self.np_a, (3, 3))
        dcr = dc.expand(self.dc_a, dc.array([3, 3]).asTypeInt())
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()