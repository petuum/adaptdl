import worker
import ray.autoscaler.sdk as sdk
import ray


class MockedRunAdaptDL:
    @staticmethod
    def options(*args, **kwargs):
        class InnerClass:
            def __init__(self):
                self.called = False
                self.args = None
                self.kwargs = None

            def remote(self, *args, **kwargs):
                self.called = True
                self.args = args
                self.kwargs = kwargs
                return self
        return InnerClass()


worker.run_adaptdl = MockedRunAdaptDL


def mocked_nodes():
    current_ip = ray.services.get_node_ip_address()
    return [
        {"NodeManagerAddress": current_ip,
         "alive": True,
         "Resources": "some value"},
        {"NodeManagerAddress": "some ip address",
         "alive": True},
        {"NodeManagerAddress": "some other ip address",
         "alive": False,
         "Resources": "some value"},
        {"NodeManagerAddress": "one last ip address",
         "alive": True,
         "Resources": {"CPU": 2}}]


worker.bundles = None


def mocked_sdk_request(bundles=None, *args, **kwargs):
    worker.bundles = bundles


sdk.request_resources = mocked_sdk_request
ray.nodes = mocked_nodes


import controller # noqa

controller.FULL_RESCALE_TIMEOUT = 4

from controller import Cluster, RayAdaptDLJob # noqa


async def test_adaptdl_job_create_workers():
    job = RayAdaptDLJob({"CPU": 2}, 0, 0)

    async def mocked_handle_worker_failure(tasks):
        pass

    job._handle_worker_failure = mocked_handle_worker_failure
    await job._create_workers([
        "virtual_node_0", "virtual_node_1", "virtual_node_2"])
    assert len(job._worker_tasks) == 3
    for i in range(3):
        print(type(job._worker_tasks[i]))
        assert job._worker_tasks[i].called


async def test_adaptdl_job_update_workers():
    job = RayAdaptDLJob({"CPU": 2}, 0, 3)
    allocation = ["virtual_node_0", "virtual_node_1", "virtual_node_2"]

    async def mocked_force_worker_checkpoint():
        print("forcing checkpoint")
        job._checkpoint_received = True
        job._checkpoint = 3

    async def mocked_create_workers(allocation):
        job._workers_created = True
        job._worker_ips = allocation

    job.force_worker_checkpoint = mocked_force_worker_checkpoint
    job._workers_created = False
    job._worker_ips = None
    job._create_workers = mocked_create_workers

    await job.update_workers(
        allocation, force_checkpoint=True, force_update=False)
    assert job._checkpoint == 3
    assert job._workers_created
    assert job._worker_ips == allocation

    job._checkpoint = 0
    await job.update_workers(
        allocation, force_checkpoint=False, force_update=True)
    assert job._checkpoint == 0
    assert job._workers_created
    assert job._worker_ips == allocation

    job._workers_created = False
    job._workers = {i: value for i, value in enumerate(allocation)}
    await job.update_workers(
        allocation, force_checkpoint=False, force_update=False)
    assert not job._workers_created
    assert job._checkpoint == 0


def test_cluster_get_nodes():
    cluster = Cluster(None, 0)
    cluster.mark_node_for_termination("some ip address")
    nodes = cluster.get_nodes()
    assert len(nodes) == 1
    assert nodes[0]["NodeManagerAddress"] == "one last ip address"


def test_cluster_ready():
    cluster = Cluster({"CPU": 1}, 5)
    allocation_1 = ["one last ip address"]
    allocation_2 = ["one last ip address", "an ip we don't have"]
    allocation_3 = ["one last ip address", "an ip we don't have",
                    "one ip too many"]

    found, count = cluster._cluster_ready(allocation_1)
    assert found
    assert count == 1

    found, count = cluster._cluster_ready(allocation_2)
    assert found
    assert count == 2

    found, count = cluster._cluster_ready(allocation_3)
    assert not found
    assert count == 2


async def test_expand_cluster():
    cluster = Cluster({"CPU": 1}, 3)
    # Note: placeholders will have "virtual" in their name
    allocation = [
        "a virtual ip we don't have",
        "one last real ip address",
        "one virtual ip too many"]
    result = await cluster.expand_cluster({}, allocation)
    assert len(worker.bundles) == 3
    assert "one last real ip address" in result
    assert len(result) == 2

    allocation = [
        "a virtual ip we don't have",
        "one last real ip address"]
    result = await cluster.expand_cluster({}, allocation)
    assert len(worker.bundles) == 2
    assert "one last real ip address" in result
    assert len(result) == 2
