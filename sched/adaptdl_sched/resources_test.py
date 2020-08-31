# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import adaptdl_sched.resources as resources
import copy
import os


def test_discretize_resource_cpu1():
    name = "cpu"
    value = "10k"
    assert(resources._discretize_resource(name, value) == 1e7)


def test_discretize_resource_cpu2():
    name = "cpu"
    value = "100m"
    assert(resources._discretize_resource(name, value) == 100)


def test_discretize_resource_cpu3():
    name = "cpu"
    value = 1000
    assert(resources._discretize_resource(name, value) == 1e6)


def test_discretize_resource_cpu4():
    name = "cpu"
    value = "1M"
    assert(resources._discretize_resource(name, value) == 1e9)


def test_discretize_resource_cpu5():
    name = "cpu"
    value = "1.37m"
    assert(resources._discretize_resource(name, value) == 2)


def test_discretize_resource_mem1():
    name = "memory"
    value = "1T"
    assert(resources._discretize_resource(name, value) == 1e12)


def test_discretize_resource_mem2():
    name = "memory"
    value = "1Gi"
    assert(resources._discretize_resource(name, value) == 1024 ** 3)


def test_discretize_resource_mem3():
    name = "memory"
    value = "10e3"
    assert(resources._discretize_resource(name, value) == 10000)


pod_spec = {
        'containers': [{
            'image': 'foo',
            'name': 'foo',
            'command': ['foo'],
            'resources': {}
        }],
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {
            'name': 'foo'
        }
    }


def test_set_default_resources1():
    os.environ['ADAPTDL_JOB_DEFAULT_RESOURCES'] = \
        '{"limits": {"memory": 100}, "requests": {"memory": 200}}'
    pod_spec_copy = copy.deepcopy(pod_spec)
    pod_spec_copy = resources.set_default_resources(pod_spec_copy)
    res = pod_spec_copy['containers'][0]['resources']
    assert(res == {"limits": {"memory": 100}, "requests": {"memory": 200}})


def test_set_default_resources2():
    os.environ['ADAPTDL_JOB_DEFAULT_RESOURCES'] = \
        '{"limits": {"memory": 100}, "requests": {"memory": 200}}'
    pod_spec_copy = copy.deepcopy(pod_spec)
    pod_spec_copy['containers'][0]['resources'] = {"limits": {"memory": 300}}
    pod_spec_copy = resources.set_default_resources(pod_spec_copy)
    res = pod_spec_copy['containers'][0]['resources']
    assert(res == {"limits": {"memory": 300}, "requests": {"memory": 200}})


def test_set_default_resources3():
    os.environ['ADAPTDL_JOB_DEFAULT_RESOURCES'] = \
        '{"requests": {"memory": 200}}'
    pod_spec_copy = copy.deepcopy(pod_spec)
    pod_spec_copy['containers'][0]['resources'] = {"limits": {"memory": 300}}
    pod_spec_copy = resources.set_default_resources(pod_spec_copy)
    res = pod_spec_copy['containers'][0]['resources']
    assert(res == {"limits": {"memory": 300}, "requests": {"memory": 200}})


def test_set_default_resources4():
    os.environ['ADAPTDL_JOB_DEFAULT_RESOURCES'] = \
        '{"limits": {"memory": 200}}'
    pod_spec_copy = copy.deepcopy(pod_spec)
    pod_spec_copy['containers'][0]['resources'] = {"limits": {"memory": 300}}
    pod_spec_copy = resources.set_default_resources(pod_spec_copy)
    res = pod_spec_copy['containers'][0]['resources']
    assert(res == {"limits": {"memory": 300}})


pod_spec_2 = {
    'containers': [
        {
            'image': 'foo',
            'name': 'foo',
            'command': ['foo'],
            'resources': {
                'requests': {
                    'memory': 100,
                    'cpu': 1.5
                },
                'limits': {
                    'memory': 80,
                    'cpu': 1,
                    'cows': 2
                }
            }
        }, {
            'image': 'foo2',
            'name': 'foo2',
            'command': ['foo2'],
            'resources': {
                'requests': {
                    'memory': 120,
                    'cpu': 2.5
                },
                'limits': {
                    'memory': 100,
                    'cpu': 2,
                    'cows': 5
                }
            }
        }, {
            'image': 'foo3',
            'name': 'foo3',
            'command': ['foo3'],
            'resources': {
                'requests': {
                    'memory': 250,
                    'cpu': 10
                },
                'limits': {
                    'memory': 250,
                    'cpu': 10
                }
            }
        }
    ],
    'apiVersion': 'v1',
    'kind': 'Pod',
    'metadata': {
        'name': 'foo'
    }
}


def test_get_pod_requests():
    pod_spec_copy = copy.deepcopy(pod_spec_2)
    requests = resources.get_pod_requests(pod_spec_copy)
    assert(requests == {
        "pods": 1,
        "cpu": resources._discretize_resource("cpu", 14),
        "memory": resources._discretize_resource("memory", 470),
        "cows": resources._discretize_resource("cows", 7)})
